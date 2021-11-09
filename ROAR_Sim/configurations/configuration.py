def import_carla():
    from pathlib import Path
    import os, sys

    # clear path s.t. path contains no carla

    sys.path = [p for p in sys.path if "carla" not in p]
    if "carla" in sys.modules:
        del sys.modules["carla"]
    roar_sim_folder_path = Path(os.getcwd())/ "ROAR_Sim"

    if roar_sim_folder_path.exists() is False:
        roar_sim_folder_path = Path(os.getcwd()).parent / "ROAR_Sim"
    if roar_sim_folder_path.exists() is False:
        roar_sim_folder_path = Path(os.getcwd()).parent.parent / "ROAR_Sim"

    carla_version_file_path = roar_sim_folder_path / "configurations" / "carla_version.txt"
    carla_version = (carla_version_file_path.open('r').readline()).strip()
    carla_client_folder_path = roar_sim_folder_path / "carla_client"

    if sys.platform == 'darwin':
        assert False, "MacOS is currently not supported"

    carla_client_egg_file_name = f'carla-{carla_version}-py3.7-win-amd64.egg' if \
        sys.platform == "win32" else f"carla-{carla_version}-py3.6-linux-x86_64.egg"
    carla_client_egg_file_path = carla_client_folder_path / carla_client_egg_file_name
    if not carla_client_egg_file_path.is_file():
        raise FileNotFoundError(
            "Please make sure carla client distribution is installed under the "
            "carla_client directory")
    sys.path.append(carla_client_egg_file_path.as_posix())
    import carla

import_carla()

from pydantic import BaseModel, Field
from ROAR_Sim.carla_client.util.utilities import CarlaWeathers, CarlaWeather, CarlaCarColors, CarlaCarColor
from typing import Optional

class Configuration(BaseModel):
    # carla server setting
    host: str = Field(default="127.0.0.1", title="Host IP",
                      description="The IP Address of the Carla Server")
    port: int = Field(default=2000, title="Host Port",
                      description="The Port number of the Carla Server")
    timeout: float = Field(default=5.0, title="Connection timeout")

    # PyGame display setting
    width: int = Field(default=1280, title="Width of Display")
    height: int = Field(default=720, title="Height of Display")

    # # carla world settings
    carla_weather: CarlaWeather = Field(default=CarlaWeathers.SUNNY,
                                        title="Carla Weather",
                                        description="Weather Setting")

    # carla vehicle setting
    carla_vehicle_blueprint_filter: str = Field(default='vehicle.tesla.model3',
                                                title="Carla Vehicle "
                                                      "Blueprint",
                                                description="For more "
                                                            "detail, see "
                                                            "https://carla.readthedocs.io/en/0.9.9/bp_library/")
    role_name: str = Field(default="hero", title="Name",
                           description="Your name in Carla, can be used to "
                                       "correctly identify you later")
    car_color: CarlaCarColor = Field(default=CarlaCarColors.RED,
                                     title="Vehicle color",
                                     description="Your Vehicle Color")

    # main camera setting (the third person camera)
    gamma: float = Field(default=2.2, title="Gamma Correction",
                         description="Gamma Correction of the camera")

    save_semantic_segmentation: bool = Field(default=False)

    print_keyboard_hint: bool = Field(default=False)

    synchronous_mode: bool = Field(default=False, title="Run Server and Client in Synchronos mode",
                                   description="https://carla.readthedocs.io/en/0.9.9/adv_synchrony_timestep/#client-server-synchrony")
    no_rendering_mode: bool = Field(default=False,
                                    title="No Rendering at all, however you can enable save data to do a play back",
                                    description="https://carla.readthedocs.io/en/0.9.9/adv_rendering_options/")
    fixed_delta_seconds: Optional[float] = Field(
        default= None,
        title="Fixed timestep with which server and client tick",
        description="Note that for us, this will ONLY take effect when you are on sync mode. "
                    "https://carla.readthedocs.io/en/0.9.9/adv_synchrony_timestep/#client-server-synchrony")

    should_spawn_npcs: bool = Field(default=False)
    npc_config_file_path: str = Field(default="./ROAR_Sim/configurations/npc_config.json")
    carla_version: str = Field(default="0.9.9")
    should_visualize_with_pygame: bool = Field(default=True)



