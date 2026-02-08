from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SCICategoryType(str, Enum):
    PHYSICAL_HEALTH = "PhysicalHealth"
    MENTAL_HEALTH_AND_LIFESTYLE = "MentalHealthAndLifestyle"
    ACTIVITIES_AND_EQUIPMENT = "ActivitiesAndEquipment"
    SUPPORT_AND_RESOURCES = "SupportAndResources"
    SOCIAL_AND_FINANCIAL = "SocialAndFinancial"


class SCICategory(BaseModel):
    type: SCICategoryType = Field(..., alias='@type')
    name: Optional[str] = Field(default=None, alias="name")
    description: Optional[str] = Field(default=None, alias="description")
    impact: Optional[str] = Field(default=None, alias="impact")
    causally_related_to: Optional[str] = Field(default=None, alias="causally_related_to")
    ameliorates: Optional[str] = Field(default=None, alias="ameliorates")


class SCICategoriesOntologyGraph(BaseModel):
    graph: Optional[List[SCICategory]] = Field(default_factory=list, alias="@graph")
