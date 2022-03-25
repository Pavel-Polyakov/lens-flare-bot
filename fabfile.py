import configparser

from fabric.api import task, local, env, sudo, execute

config = configparser.ConfigParser()
config.read('config.ini')

env.hosts = [config['main']['host']]
image = config['docker']['image']
container_name = config['docker']['container_name']

token = config['telegram']['token']


@task(alias='n')
def notify_when_task_done():
    script = 'display notification "Done ✅" with title "Fabric" sound name "Pop"'
    local(f"osascript -e '{script}'")


@task
def build():
    local(f'docker build -t {image} -f Dockerfile .')


@task
def run():
    local(f'docker run -it --rm -e TOKEN={token} {image}')


@task(alias='brt')
def build_and_run():
    execute(build)
    execute(run)


@task
def push():
    local(f'docker push {image}')


@task
def deploy():
    commands = [
        f'docker pull {image}',
    ]

    containers = sudo("docker ps --format '{{json .Names}}'").replace('"', '').splitlines()
    if container_name in containers:
        commands.extend([
            f'docker stop {container_name}',
            f'docker rm {container_name}',
        ])

    commands.append(
        f'docker run -e TOKEN={token} -e TZ=Europe/Moscow --name={container_name} --restart=always --detach=true -t {image}'
    )

    sudo(' && '.join(commands))


@task
def stop():
    sudo(f'docker stop {container_name}')


@task(alias='bd')
def build_and_deploy():
    execute(build)
    execute(push)
    execute(deploy)
    execute(notify_when_task_done)


@task
def logs():
    sudo(f"docker logs -f {container_name}")
