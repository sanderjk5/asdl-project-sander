from abc import ABC
from itertools import combinations
from typing import Any, Dict, FrozenSet, List, Tuple, Type, Union

import libcst as cst
from libcst.metadata import CodePosition, CodeRange, ParentNodeProvider, PositionProvider, Scope, ScopeProvider
from libcst.metadata.scope_provider import (
    BuiltinAssignment,
    ComprehensionScope,
    FunctionScope,
    GlobalScope,
    QualifiedNameSource,
)
from typing_extensions import Final

from .rewriteops import AbstractRewriteOp, ReplaceCodeTextOp


class ICodeRewriteScout(cst.BatchableCSTVisitor, ABC):
    """
    Interface of a class that finds the locations where a code rewrite can be applied.
    All possible rewrites are accumulated in the `op_store` list passed in the constructor.
    The `op_metadata_store` optionally stores additional metadata.
    """

    def __init__(
        self,
        op_store: List[AbstractRewriteOp],
        op_metadata_store: List[Tuple[Type["ICodeRewriteScout"], cst.CSTNode, Any]] = None,
    ):
        super().__init__()
        self.__ops = op_store
        self.__op_metadata_store = op_metadata_store

    def add_mod_op(self, op: AbstractRewriteOp, metadata: Tuple[Type["ICodeRewriteScout"], cst.CSTNode, Any]) -> None:
        self.__ops.append(op)
        if self.__op_metadata_store is not None:
            self.__op_metadata_store.append(metadata)


class LoopStatementRewriteScout(ICodeRewriteScout):
    """The added rewrite scout which introduces a bug by replacing a statement in a loop."""

    METADATA_DEPENDENCIES = (PositionProvider,)

    LoopStmts: Final = frozenset(
        {
            cst.Continue,
            cst.Break,
        }
    )

    DISABLED_LOOPSTMTS = frozenset()

    def visit_Continue(self, node: cst.Continue):
        assert isinstance(node, cst.Continue)

        op_range: CodeRange = self.get_metadata(PositionProvider, node)
        replace_op = ReplaceCodeTextOp(op_range, "break")
        self.add_mod_op(replace_op, (LoopStatementRewriteScout, node, None))
    
    def visit_Break(self, node: cst.Break):
        assert isinstance(node, cst.Break)

        op_range: CodeRange = self.get_metadata(PositionProvider, node)
        replace_op = ReplaceCodeTextOp(op_range, "continue")
        self.add_mod_op(replace_op, (LoopStatementRewriteScout, node, None))