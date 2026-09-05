from marker.schema import BlockTypes
from marker.schema.blocks import Block


class Bibliography(Block):
    block_type: BlockTypes = BlockTypes.Bibliography
    html: str | None = None
    block_description: str = "A bibliography or list of references."

    def assemble_html(
        self, document, child_blocks, parent_structure, block_config=None
    ):
        if self.ignore_for_output:
            return ""

        if self.html:
            return super().handle_html_output(
                document, child_blocks, parent_structure, block_config
            )

        template = super().assemble_html(
            document, child_blocks, parent_structure, block_config
        )
        template = template.replace("\n", " ")
        return f"<p block-type='{self.block_type}'>{template}</p>"
