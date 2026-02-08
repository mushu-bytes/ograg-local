from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SCIEntityType(str, Enum):
    CONDITION = "SCICondition"
    SPECIALIST = "Specialist"
    TREATMENT = "Treatment"
    PRODUCT = "Product"
    STAGE = "Stage"


class Specialist(BaseModel):
    type: SCIEntityType = Field(SCIEntityType.SPECIALIST, alias='@type')
    specialist_name: str = Field(..., alias="specialist_name")


class Treatment(BaseModel):
    type: SCIEntityType = Field(SCIEntityType.TREATMENT, alias='@type')
    treatment_name: str = Field(..., alias="treatment_name")
    treatment_phase: Optional[str] = Field(default=None, alias="treatment_phase")


class Product(BaseModel):
    type: SCIEntityType = Field(SCIEntityType.PRODUCT, alias='@type')
    product_name: str = Field(..., alias="product_name")
    product_category: Optional[str] = Field(default=None, alias="product_category")


class Stage(BaseModel):
    type: SCIEntityType = Field(SCIEntityType.STAGE, alias='@type')
    stage_name: str = Field(..., alias="stage_name")


class SCICondition(BaseModel):
    type: SCIEntityType = Field(SCIEntityType.CONDITION, alias='@type')
    condition_name: str = Field(..., alias="condition_name")
    symptoms_description: Optional[str] = Field(default=None, alias="symptoms_description")
    how_to_prevent: Optional[str] = Field(default=None, alias="how_to_prevent")
    how_to_manage: Optional[str] = Field(default=None, alias="how_to_manage")
    impact_description: Optional[str] = Field(default=None, alias="impact_description")
    treated_by_specialists: Optional[List[Specialist]] = Field(default=None, alias="treated_by_specialists")
    treatment_options: Optional[List[Treatment]] = Field(default=None, alias="treatment_options")
    helpful_products: Optional[List[Product]] = Field(default=None, alias="helpful_products")
    relevant_to_stage: Optional[List[Stage]] = Field(default=None, alias="relevant_to_stage")
    home_modifications_needed: Optional[str] = Field(default=None, alias="home_modifications_needed")


class SCIOntologyGraph(BaseModel):
    graph: List[SCICondition | Specialist | Treatment | Product | Stage] = Field(default_factory=list, alias="@graph")
