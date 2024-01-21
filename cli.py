import uvicorn
from endaaman.cli2 import BaseCLI

class CLI(BaseCLI):
    class CommonArgs(BaseCLI.CommonArgs):
        port: int = 8000

    def run_dev(self, a):
        uvicorn.run('robiemon_server:app', host='0.0.0.0', port=a.port, reload=True)

    def run_prod(self, a):
        uvicorn.run('robiemon_server:app', host='0.0.0.0', port=a.port)

    def run_migrate(self, a):
        # Base.metadata.create_all(bind=engine)
        pass

if __name__ == '__main__':
    cli = CLI()
    cli.run()
