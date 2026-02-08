from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SCISubcategoryType(str, Enum):
    MOBILITY = "Mobility"
    AUTONOMIC_HEALTH = "AutonomicHealth"
    PEDIATRICS = "Pediatrics"
    WOMEN = "Women"
    OTHER = "Other"
    MENTAL_HEALTH = "MentalHealth"
    LIFESTYLE = "Lifestyle"
    LIFE_WITH_SCI = "LifeWithSCI"
    AGING = "Aging"
    SEXUAL_HEALTH = "SexualHealth"
    NUTRITION_AND_WELLNESS = "NutritionAndWellness"
    SPORTS_RECREATION_AND_FITNESS = "SportsRecreationAndFitness"
    TOOLS_AND_TECHNOLOGY = "ToolsAndTechnology"
    HOME_MODIFICATIONS_AND_ADAPTIVE_EQUIPMENT = "HomeModificationsAndAdaptiveEquipment"
    MOBILITY_EQUIPMENT = "MobilityEquipment"
    ADAPTIVE_DRIVING = "AdaptiveDriving"
    ADVOCACY = "Advocacy"
    CAREGIVER = "Caregiver"
    SUPPORT_NETWORK = "SupportNetwork"
    ALTERNATIVE_REHABILITATION_SERVICES = "AlternativeRehabilitationServices"
    PARENTING_WITH_SCI = "ParentingWithSCI"
    FINANCES_EMPLOYMENT = "FinancesEmployment"
    RELATIONSHIPS = "Relationships"
    TRAVEL = "Travel"
    SOCIAL_CULTURAL_AND_ENVIRONMENT_BARRIERS = "SocialCulturalAndEnvironmentBarriers"
    MISCELLANEOUS = "Miscellaneous"


class SCISubcategory(BaseModel):
    type: SCISubcategoryType = Field(..., alias='@type')
    name: Optional[str] = Field(default=None, alias="name")
    description: Optional[str] = Field(default=None, alias="description")
    impact: Optional[str] = Field(default=None, alias="impact")
    causally_related_to: Optional[str] = Field(default=None, alias="causally_related_to")
    ameliorates: Optional[str] = Field(default=None, alias="ameliorates")


class SCISubcategoriesOntologyGraph(BaseModel):
    graph: Optional[List[SCISubcategory]] = Field(default_factory=list, alias="@graph")
