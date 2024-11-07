"""
GeoSpace
--------
"""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pyproj
from mesa.agent import AgentSet

from mesa_geo.geo_base import GeoBase
from mesa_geo.geoagent import GeoAgent
from mesa_geo.raster_layers import ImageLayer, RasterLayer


class GeoSpace(GeoBase):
    """
    Space used to add a geospatial component to a model.
    """

    def __init__(self, crs="epsg:3857", *, warn_crs_conversion=True):
        """
        Create a GeoSpace for GIS enabled mesa modeling.

        :param crs: The coordinate reference system of the GeoSpace.
            If `crs` is not set, epsg:3857 (Web Mercator) is used as default.
            However, this system is only accurate at the equator and errors
            increase with latitude.
        :param warn_crs_conversion: Whether to warn when converting layers and
            GeoAgents of different crs into the crs of GeoSpace. Default to
            True.
        """
        super().__init__(crs)
        self._transformer = pyproj.Transformer.from_crs(
            crs_from=self.crs, crs_to="epsg:4326", always_xy=True
        )
        self.warn_crs_conversion = warn_crs_conversion
        self._static_layers = []
        self._total_bounds = None  # [min_x, min_y, max_x, max_y]

    def to_crs(self, crs, inplace=False) -> GeoSpace | None:
        super()._to_crs_check(crs)

        if inplace:
            for agent in self.agents:
                agent.to_crs(crs, inplace=True)
            for layer in self.layers:
                layer.to_crs(crs, inplace=True)
        else:
            geospace = GeoSpace(
                crs=self.crs.to_string(), warn_crs_conversion=self.warn_crs_conversion
            )
            for agent in self.agents:
                geospace.add_agents(agent.to_crs(crs, inplace=False))
            for layer in self.layers:
                geospace.add_layer(layer.to_crs(crs, inplace=False))
            return geospace

    @property
    def transformer(self):
        """
        Return the pyproj.Transformer that transforms the GeoSpace into
        epsg:4326. Mainly used for GeoJSON serialization.
        """
        return self._transformer

    @property
    def agents(self) -> AgentSet[GeoAgent]:
        """
        Return a list of all agents in the Geospace.
        """
        return self._grid.agents

    @property
    def layers(self) -> list[ImageLayer | RasterLayer | gpd.GeoDataFrame]:
        """
        Return a list of all layers in the Geospace.
        """
        return self._static_layers

    @property
    def total_bounds(self) -> np.ndarray | None:
        """
        Return the bounds of the GeoSpace in [min_x, min_y, max_x, max_y] format.
        """
        if self._total_bounds is None and len(self.layers) > 0:
            for layer in self.layers:
                self._update_bounds(layer.total_bounds)
        return self._total_bounds

    def _update_bounds(self, new_bounds: np.ndarray) -> None:
        if new_bounds is not None:
            if self._total_bounds is not None:
                new_min_x = min(self.total_bounds[0], new_bounds[0])
                new_min_y = min(self.total_bounds[1], new_bounds[1])
                new_max_x = max(self.total_bounds[2], new_bounds[2])
                new_max_y = max(self.total_bounds[3], new_bounds[3])
                self._total_bounds = np.array(
                    [new_min_x, new_min_y, new_max_x, new_max_y]
                )
            else:
                self._total_bounds = new_bounds

    @property
    def __geo_interface__(self):
        """
        Return a GeoJSON FeatureCollection.
        """
        features = [a.__geo_interface__() for a in self.agents]
        return {"type": "FeatureCollection", "features": features}

    def add_layer(self, layer: ImageLayer | RasterLayer | gpd.GeoDataFrame) -> None:
        """Add a layer to the Geospace.

        :param ImageLayer | RasterLayer | gpd.GeoDataFrame layer: The layer to add.
        """
        if not self.crs.is_exact_same(layer.crs):
            if self.warn_crs_conversion:
                warnings.warn(
                    f"Converting {layer.__class__.__name__} from crs {layer.crs.to_string()} "
                    f"to the crs of {self.__class__.__name__} - {self.crs.to_string()}. "
                    "Please check your crs settings if this is unintended, or set `GeoSpace.warn_crs_conversion` "
                    "to `False` to suppress this warning message.",
                    UserWarning,
                    stacklevel=2,
                )
            layer.to_crs(self.crs, inplace=True)
        self._total_bounds = None
        self._static_layers.append(layer)

    def _check_agent(self, agent):
        if hasattr(agent, "geometry"):
            if not self.crs.is_exact_same(agent.crs):
                if self.warn_crs_conversion:
                    warnings.warn(
                        f"Converting {agent.__class__.__name__} from crs {agent.crs.to_string()} "
                        f"to the crs of {self.__class__.__name__} - {self.crs.to_string()}. "
                        "Please check your crs settings if this is unintended, or set `GeoSpace.warn_crs_conversion` "
                        "to `False` to suppress this warning message.",
                        UserWarning,
                        stacklevel=2,
                    )
                agent.to_crs(self.crs, inplace=True)
        else:
            raise AttributeError("GeoAgents must have a geometry attribute")

    def add_agents(self, agents):
        """Add a list of GeoAgents to the Geospace.

        GeoAgents must have a geometry attribute. This function may also be called
        with a single GeoAgent.

        :param agents: A list of GeoAgents or a single GeoAgent to be added into GeoSpace.
        :raises AttributeError: If the GeoAgents do not have a geometry attribute.
        """
        if isinstance(agents, GeoAgent):
            agent = agents
            self._check_agent(agent)
        else:
            for agent in agents:
                self._check_agent(agent)
        self._total_bounds = None

    def get_neighbors_within_distance(
        self, agent, distance, center=False, relation="intersects"
    ):
        """Return a list of agents within `distance` of `agent`.

        Distance is measured as a buffer around the agent's geometry,
        set center=True to calculate distance from center.
        """
        yield from self._agent_layer.get_neighbors_within_distance(
            agent, distance, center, relation
        )

    def agents_at(self, pos):
        """
        Return a list of agents at given pos.
        """
        return self._agent_layer.agents_at(pos)

    def distance(self, agent_a, agent_b):
        """
        Return distance of two agents.
        """
        return self._agent_layer.distance(agent_a, agent_b)

    def get_neighbors(self, agent):
        """
        Get (touching) neighbors of an agent.
        """
        return self._agent_layer.get_neighbors(agent)

    def get_agents_as_GeoDataFrame(self, agent_cls=GeoAgent) -> gpd.GeoDataFrame:
        """
        Extract GeoAgents as a GeoDataFrame.

        :param agent_cls: The class of the GeoAgents to extract. Default is `GeoAgent`.
        :return: A GeoDataFrame of the GeoAgents.
        :rtype: gpd.GeoDataFrame
        """

        return self._agent_layer.get_agents_as_GeoDataFrame(agent_cls)
