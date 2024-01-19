import uvicorn
from endaaman.cli2 import BaseCLI

from robiemon_server.app import app

class CLI(BaseCLI):
    class CommonArgs(BaseCLI.CommonArgs):
        port: int = 8000

    def run_dev(self, a):
        uvicorn.run(app, host='0.0.0.0', port=a.port, reload=True)

    def run_prod(self, a):
        uvicorn.run(app, host='0.0.0.0', port=a.port)

if __name__ == '__main__':
    cli = CLI()
    cli.run()
