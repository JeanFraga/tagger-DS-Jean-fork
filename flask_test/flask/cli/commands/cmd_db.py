import click

from sqlalchemy_utils import database_exists, create_database

from snakeeyes.app import create_app
from snakeeyes.extensions import db

# Create an app context for the database connection.
app = create_app()
db.app = app


@click.group()
def cli():
    """ Run PostgreSQL related tasks. """
    pass


@click.command()
@click.option('--with-testdb/--no-with-testdb', default=False,
              help='Create a test db too?')
def init(with_testdb):
    db.drop_all()
    db.create_all()

    return None


@click.command()
@click.option('--with-testdb/--no-with-testdb', default=False,
              help='Create a test db too?')
@click.pass_context
def reset(ctx):
    ctx.invoke(init)

    return None


cli.add_command(init)
cli.add_command(reset)
