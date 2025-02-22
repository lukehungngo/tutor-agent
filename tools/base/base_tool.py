from typing import Any
from langchain.tools import Tool


class ToolFailure(Exception):
    """Base class for tool failures."""

    pass


class CriticalToolFailure(ToolFailure):
    """Exception for critical tool failures that should cascade."""

    pass


class NonCriticalToolFailure(ToolFailure):
    """Exception for non-critical tool failures that can be handled locally."""

    pass


class TutorBaseTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        return_direct: bool = False,
        failure_mode: type[ToolFailure] = NonCriticalToolFailure,
        **kwargs: Any,
    ):
        super().__init__(
            name=name, description=description, return_direct=return_direct, **kwargs
        )
        self.failure_mode = failure_mode

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().invoke(*args, **kwargs)
        except Exception as e:
            if self.failure_mode == CriticalToolFailure:
                raise
            raise NonCriticalToolFailure(str(e))
        pass

    def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().ainvoke(*args, **kwargs)
        except Exception as e:
            if isinstance(e, CriticalToolFailure):
                raise
            raise NonCriticalToolFailure(str(e))
        pass
