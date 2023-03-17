"""Command Line Interface of the x-annotate package."""

import click
from xannotate.frontend import FrontendClient

DEBUG = True


########################################################################################################################
# HELPER
########################################################################################################################
def context2client(_ctx):
    context = {k: v for k, v in _ctx.obj.items()}
    return context["client"]


########################################################################################################################
# CLI
########################################################################################################################
@click.group()
@click.option("--config_file", default=None, type=str, help="[str] relative path to config file")
@click.option("--project_file", default=None, type=str, help="[str] relative path to project file")
@click.pass_context
def xannotate(ctx, **kwargs_optional):
    ctx.ensure_object(dict)

    # config file and project file
    kwargs = {k: v for k, v in kwargs_optional.items() if v is not None}
    config_file = kwargs["config_file"] if "config_file" in kwargs.keys() else ".config.ini"
    project_file = kwargs["project_file"] if "project_file" in kwargs.keys() else ".project.ini"

    # establish connection
    client = FrontendClient(config_file, project_file)

    # context
    context = {
        "client": client,
    }
    ctx.obj = context


########################################################################################################################
# MAIN COMMANDS
########################################################################################################################
@xannotate.command(name="project")
@click.pass_context
def project(ctx):
    """
    show project details
    """
    client = context2client(ctx)
    print(client.project)


@xannotate.command(name="config")
@click.pass_context
def config(ctx):
    """
    show config details
    """
    client = context2client(ctx)
    print(client.config)


@xannotate.command(name="status")
@click.pass_context
def status(ctx):
    """
    check annotation status
    """
    client = context2client(ctx)
    client.overview()


@xannotate.command(name="assign")
@click.pass_context
def assign(ctx):
    """
    assign data to annotators
    """
    client = context2client(ctx)
    client.first_step()
    client.convert2doccano(batch_stage="A")
    client.split()
    client.doccano_post(batch_stage="A")


@xannotate.command(name="merge")
@click.pass_context
def merge(ctx):
    """
    merge individual annotations
    """
    client = context2client(ctx)
    client.doccano_get(batch_stage="B")
    client.local_merge()
    client.doccano_post(batch_stage="B")


@xannotate.command(name="finish")
@click.pass_context
def finish(ctx):
    """
    download final annotations and finish
    """
    client = context2client(ctx)
    client.doccano_get(batch_stage="X")
    client.convert2standard(batch_stage="X")
    client.last_step()


########################################################################################################################
# HIDDEN COMMANDS
########################################################################################################################
@xannotate.command(name="firststep", hidden=True)
@click.pass_context
def first_step(ctx):
    client = context2client(ctx)
    client.first_step()


@xannotate.command(name="laststep", hidden=True)
@click.pass_context
def last_step(ctx):
    client = context2client(ctx)
    client.last_step()


@xannotate.command(name="convert2doccano", hidden=True)
@click.pass_context
def convert2doccano(ctx):
    client = context2client(ctx)
    client.convert2doccano(batch_stage="A")


@xannotate.command(name="split", hidden=True)
@click.pass_context
def split(ctx):
    client = context2client(ctx)
    client.split()


@xannotate.command(name="post", hidden=True)
@click.pass_context
@click.argument("batch_stage")
def post(ctx, batch_stage: str):
    client = context2client(ctx)
    client.doccano_post(batch_stage=batch_stage)


@xannotate.command(name="get", hidden=True)
@click.argument("batch_stage")
@click.pass_context
def get(ctx, batch_stage: str):
    client = context2client(ctx)
    # batch_stage_previous = BATCH_STAGE_PREVIOUS_MAPPING[batch_stage]
    # batch_id = BatchIdTracker.get_batch_id(batch_stage=batch_stage_previous)
    client.doccano_get(batch_stage=batch_stage)


@xannotate.command(name="local_merge", hidden=True)
@click.pass_context
def local_merge(ctx):
    client = context2client(ctx)
    client.local_merge()


@xannotate.command(name="convert2standard", hidden=True)
@click.pass_context
def convert2standard(ctx):
    client = context2client(ctx)
    client.convert2standard(batch_stage="X")
