import logging
import os
import tempfile
import traceback
import uuid
from queue import Queue

import telebot

from lens_flare import AutoLensFlare, StepByStepLensFlare, Config

bot = telebot.TeleBot(os.getenv('TOKEN'))
logger = logging.getLogger(__name__)


class Commands:
    SKIP = 'SKIP'
    STAY = 'STAY'

    OK = 'OK'
    EDIT = 'EDIT'


def edit_step_by_step(message, content):
    messages = [message]
    messages.append(bot.send_message(message.chat.id,
                                     'we will do it step by step and on each iteration i '
                                     'will ask you what to do with step'))

    with tempfile.TemporaryDirectory() as tmp_dir:

        new_file_name = tmp_dir + uuid.uuid4().hex
        with open(new_file_name, 'wb') as new_file:
            new_file.write(content)

        output_name = new_file_name + '_output.jpg'
        cfg = Config(
            src_file=new_file_name,
            out_file=output_name,
            star_file='star.png',
        )

        markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
        markup.add(telebot.types.KeyboardButton(Commands.SKIP))
        markup.add(telebot.types.KeyboardButton(Commands.STAY))
        # TODO: add STOP

        iterator = StepByStepLensFlare(cfg, logger).process()

        queue = Queue()

        skip = None
        out_file = None

        while True:
            try:
                out_file = iterator.send(skip)
            except StopIteration:
                break

            with open(out_file, 'rb') as f:
                sent = bot.send_photo(chat_id=message.chat.id, photo=f, reply_to_message_id=message.message_id,
                                      reply_markup=markup)

                def make_handler(sent_message):
                    def handle_response(reply_message):
                        bot.delete_message(message.chat.id, sent_message.id)
                        bot.delete_message(message.chat.id, reply_message.id)
                        if reply_message.text == Commands.SKIP:
                            queue.put(True)
                        else:
                            queue.put(False)

                    return handle_response

                bot.register_next_step_handler(sent, make_handler(sent))
                skip = queue.get()

        for m in messages:
            bot.delete_message(message.chat.id, m.id)

        if out_file is not None:
            with open(out_file, 'rb') as f:
                bot.send_photo(chat_id=message.chat.id, photo=f)


def request_feedback(message, content):
    markup = telebot.types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.add(telebot.types.KeyboardButton(Commands.OK))
    markup.add(telebot.types.KeyboardButton(Commands.EDIT))

    def make_handler(sent_message, content):
        def handle_response(reply_message):
            bot.delete_message(message.chat.id, sent_message.id)
            bot.delete_message(message.chat.id, reply_message.id)
            if reply_message.text == Commands.EDIT:
                bot.delete_message(message.chat.id, message.id)
                edit_step_by_step(bot.send_message(message.chat.id, 'starting edit mode...'), content)
            else:
                logger.error('unexpected state')

        return handle_response

    sent = bot.reply_to(message, 'choose what to do', reply_markup=markup)
    bot.register_next_step_handler(sent, make_handler(sent, content))


@bot.message_handler(content_types=['photo'])
def handler(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            new_file_name = tmp_dir + message.photo[-1].file_id
            with open(new_file_name, 'wb') as new_file:
                new_file.write(downloaded_file)

            output_name = new_file_name + '_output.jpg'

            cfg = Config(
                src_file=new_file_name,
                out_file=output_name,
                star_file='star.png',
            )

            AutoLensFlare(cfg, logger).process()

            with open(output_name, 'rb') as f:
                sent = bot.send_photo(chat_id=message.chat.id, photo=f,
                                      reply_to_message_id=message.message_id)

            request_feedback(sent, downloaded_file)

    except Exception as ex:
        traceback.print_exc()
        bot.reply_to(message, f'Failed ({ex})')


if __name__ == '__main__':
    bot.polling()
