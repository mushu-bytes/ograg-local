from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class VariableInstanceTypes(str, Enum):
    SEX = "Sex"
    PEG = "PEG"
    SLEEP_DISTURBANCE = "SleepDisturbance"
    DEPRESSION = "Depression"
    ANXIETY = "Anxiety"
    OBESITY = "Obesity"
    ALCOHOL = "Alcohol"
    FEAR_AVOIDANCE = "FearAvoidance"
    CATASTROPHIZING = "Catastrophizing"
    CCI = "CCI"
    EDUCATION = "Education"
    FINANCIAL_LEVEL = "FinancialLevel"
    AGE = "Age"
    SMOKING = "Smoking"
    VARIABLE = "Variable"

class CLBPVariableInstance(BaseModel):
    type: VariableInstanceTypes = Field(..., alias='@type')
    name: Optional[str] = Field(default=None, alias="name")
    description: Optional[str] = Field(default=None, alias="description")
    has_causes: Optional[str] = Field(default=None, alias="hasCauses")
    has_associations: Optional[str] = Field(default=None, alias="hasAssociations")
    has_temporal_causes: Optional[str] = Field(default=None, alias="hasTemporalCauses")
    has_mediators: Optional[str] = Field(default=None, alias="hasMediators")
    evidence: Optional[str] = Field(default=None, alias="evidence")

class CLBPFullCausalVariables(BaseModel):
    graph: Optional[List[CLBPVariableInstance]] = Field(default_factory=list, alias="@graph")