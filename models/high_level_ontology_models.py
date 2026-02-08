from pydantic import BaseModel, Field
from typing import List, Optional

class CLBPVariable(BaseModel):
    type: str = Field(default="Variable", alias='@type')
    name: Optional[str] = Field(default=None, alias="name")
    description: Optional[str] = Field(default=None, alias="description")
    has_causes: Optional[str] = Field(default=None, alias="hasCauses")
    has_associations: Optional[str] = Field(default=None, alias="hasAssociations")
    has_temporal_causes: Optional[str] = Field(default=None, alias="hasTemporalCauses")
    has_mediators: Optional[str] = Field(default=None, alias="hasMediators")
    evidence: Optional[str] = Field(default=None, alias="evidence")


class CLBPCausalVariable(BaseModel):
    graph: Optional[List[CLBPVariable]] = Field(default_factory=list, alias="@graph")