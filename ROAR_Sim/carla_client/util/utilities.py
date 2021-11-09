from pydantic import BaseModel, Field
import carla
from pathlib import Path


class CarlaCarColor(BaseModel):
    r: int = Field(default=0)
    g: int = Field(default=0)
    b: int = Field(default=0)
    a: int = Field(default=255)

    def to_string(self, *args, **kwargs):
        return f"{self.r},{self.g},{self.b}"


class CarlaCarColors:
    RED: CarlaCarColor = CarlaCarColor(r=255, g=0, b=0)
    BLUE: CarlaCarColor = CarlaCarColor(r=0, g=0, b=255)
    GREEN: CarlaCarColor = CarlaCarColor(r=0, g=255, b=0)
    BLACK: CarlaCarColor = CarlaCarColor(r=0, g=0, b=0)
    WHITE: CarlaCarColor = CarlaCarColor(r=255, g=250, b=250)
    GREY: CarlaCarColor = CarlaCarColor(r=211, g=211, b=211)


class CarlaWeather(BaseModel):
    """
    Default weather is sunny
    """

    cloudiness: float = Field(default=10)
    precipitation: float = Field(default=0)
    precipitation_deposits: float = Field(default=0)
    wind_intensity: float = Field(default=0)
    sun_azimuth_angle: float = Field(default=90)
    sun_altitude_angle: float = Field(default=90)
    fog_density: float = Field(default=0)
    fog_distance: float = Field(default=0)
    wetness: float = Field(default=0)

    def to_carla_weather_params(self):
        return carla.WeatherParameters(
            cloudiness=self.cloudiness,
            precipitation=self.precipitation,
            precipitation_deposits=self.precipitation_deposits,
            wind_intensity=self.wind_intensity,
            sun_azimuth_angle=self.sun_azimuth_angle,
            sun_altitude_angle=self.sun_altitude_angle,
            fog_density=self.fog_density,
            fog_distance=self.fog_distance,
            wetness=self.wetness,
        )


class CarlaWeathers:
    SUNNY = CarlaWeather()


def get_actor_display_name(actor, truncate=250):
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "\u2026") if len(name) > truncate else name


def create_dir_if_not_exist(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
