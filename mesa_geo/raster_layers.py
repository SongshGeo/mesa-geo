"""
Raster Layers
-------------
"""

from __future__ import annotations

import copy
import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Iterator
from functools import cached_property
from typing import TypeVar

import numpy as np
import rasterio as rio
from affine import Affine
from mesa import Model
from mesa.experimental.cell_space.cell import Cell, CellCollection
from mesa.experimental.cell_space.grid import (
    Grid,
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.space import Coordinate, PropertyLayer, accept_tuple_argument
from numpy.typing import NDArray
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)

from mesa_geo.geo_base import GeoBase

T = TypeVar("T", bound=Cell)


class RasterBase(GeoBase):
    """
    Base class for raster layers.
    """

    _width: int
    _height: int
    _transform: Affine
    _total_bounds: np.ndarray  # [min_x, min_y, max_x, max_y]

    def __init__(self, width, height, crs, total_bounds):
        """
        Initialize a raster base layer.

        :param width: Width of the raster base layer.
        :param height: Height of the raster base layer.
        :param crs: Coordinate reference system of the raster base layer.
        :param total_bounds: Bounds of the raster base layer in [min_x, min_y, max_x, max_y] format.
        """

        super().__init__(crs)
        self._width = width
        self._height = height
        self._total_bounds = total_bounds
        self._update_transform()

    @property
    def width(self) -> int:
        """
        Return the width of the raster base layer.

        :return: Width of the raster base layer.
        :rtype: int
        """

        return self._width

    @width.setter
    def width(self, width: int) -> None:
        """
        Set the width of the raster base layer.

        :param int width: Width of the raster base layer.
        """

        self._width = width
        self._update_transform()

    @property
    def height(self) -> int:
        """
        Return the height of the raster base layer.

        :return: Height of the raster base layer.
        :rtype: int
        """

        return self._height

    @height.setter
    def height(self, height: int) -> None:
        """
        Set the height of the raster base layer.

        :param int height: Height of the raster base layer.
        """

        self._height = height
        self._update_transform()

    @property
    def total_bounds(self) -> np.ndarray | None:
        return self._total_bounds

    @total_bounds.setter
    def total_bounds(self, total_bounds: np.ndarray) -> None:
        """
        Set the bounds of the raster base layer in [min_x, min_y, max_x, max_y] format.

        :param np.ndarray total_bounds: Bounds of the raster base layer in [min_x, min_y, max_x, max_y] format.
        """

        self._total_bounds = total_bounds
        self._update_transform()

    @property
    def transform(self) -> Affine:
        """
        Return the affine transformation of the raster base layer.

        :return: Affine transformation of the raster base layer.
        :rtype: Affine
        """

        return self._transform

    @property
    def resolution(self) -> tuple[float, float]:
        """
        Returns the (width, height) of a cell in the units of CRS.

        :return: Width and height of a cell in the units of CRS.
        :rtype: Tuple[float, float]
        """

        a, b, _, d, e, _, _, _, _ = self.transform
        return math.sqrt(a**2 + d**2), math.sqrt(b**2 + e**2)

    def _update_transform(self) -> None:
        self._transform = rio.transform.from_bounds(
            *self.total_bounds, width=self.width, height=self.height
        )

    def to_crs(self, crs, inplace=False) -> RasterBase | None:
        raise NotImplementedError

    def out_of_bounds(self, pos: Coordinate) -> bool:
        """
        Determines whether position is off the grid.

        :param Coordinate pos: Position to check.
        :return: True if position is off the grid, False otherwise.
        :rtype: bool
        """

        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height


class RasterLayer(RasterBase):
    """
    Some methods in `RasterLayer` are copied from `mesa.space.Grid`, including:

    __getitem__
    __iter__
    coord_iter
    iter_neighborhood
    get_neighborhood
    iter_neighbors
    get_neighbors  # copied and renamed to `get_neighboring_cells`
    out_of_bounds  # copied into `RasterBase`
    iter_cell_list_contents
    get_cell_list_contents

    Methods from `mesa.space.Grid` that are not copied over:

    torus_adj
    neighbor_iter
    move_agent
    place_agent
    _place_agent
    remove_agent
    is_cell_empty
    move_to_empty
    find_empty
    exists_empty_cells

    Another difference is that `mesa.space.Grid` has `self.grid: List[List[Agent | None]]`,
    whereas it is `self.cells: List[List[Cell]]` here in `RasterLayer`.
    """

    def __init__(
        self,
        width,
        height,
        crs,
        total_bounds,
        model,
        cell_cls: type[Cell] = Cell,
        moore: bool = False,
    ):
        RasterBase.__init__(self, width, height, crs, total_bounds)
        self.model = model
        self.cell_cls = cell_cls
        self._setup_grid(width, height, moore=moore, cell_cls=cell_cls)

    def _setup_grid(
        self,
        width: int,
        height: int,
        moore: bool,
        cell_cls: type[Cell],
        **kwargs,
    ):
        grid_cls = OrthogonalMooreGrid if moore else OrthogonalVonNeumannGrid
        self._grid: Grid = grid_cls(
            dimensions=(width, height),
            cell_klass=cell_cls,
            **kwargs,
        )
        self._moore = moore

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the raster layer."""
        return self.height, self.width

    @property
    def grid(self) -> Grid | None:
        """Return the grid of the raster layer."""
        return getattr(self, "_grid", None)

    @property
    def cells(self) -> CellCollection:
        """Return all cells in the raster layer."""
        return self.grid.all_cells

    @cached_property
    def array_cells(self) -> NDArray[T]:
        """Return all cells in the raster layer as a 2D numpy array."""
        array = np.empty(shape=self.shape, dtype=object)
        for cell in self.cells:
            x, y = cell.coordinate
            array[self.height - y - 1, x] = cell
        return array

    @property
    def attributes(self) -> list[str]:
        """Return the names of all attributes in the raster layer."""
        return list(self.grid.property_layers.keys())

    def __getitem__(self, index) -> Cell | CellCollection:
        """
        Access contents from the grid.
        """
        selected = self.array_cells[index]
        return (
            selected
            if isinstance(selected, Cell)
            else CellCollection(selected.flatten())
        )

    def __iter__(self) -> Iterator[Cell]:
        """
        Create an iterator that chains the rows of the cells together
        as if it is one list
        """
        return itertools.chain(self.cells)

    def coord_iter(self) -> Iterator[tuple[Cell, int, int]]:
        """
        An iterator that returns coordinates as well as cell contents.
        """

        for row in range(self.width):
            for col in range(self.height):
                yield self.cells[row, col], row, col  # cell, x, y

    def add_property(
        self,
        data: np.ndarray | float | int | bool,
        attr_name: str,
        add_to_cells: bool = True,
    ) -> None:
        """Add a property layer to the grid."""
        if isinstance(data, np.ndarray):
            if data.shape != (self.height, self.width):
                raise ValueError(
                    f"Data shape does not match raster shape. "
                    f"Expected {(self.height, self.width)}, received {data.shape}."
                )
        else:
            data = np.full((self.height, self.width), data)
        property_layer = PropertyLayer(
            attr_name,
            self.width,
            self.height,
            default_value=np.nan,
        )
        property_layer.data = data
        self.grid.add_property_layer(property_layer, add_to_cells)

    def apply_raster(self, data: np.ndarray, attr_name: str | None = None) -> None:
        """
        Apply raster data to the cells.

        :param np.ndarray data: 2D numpy array with shape (1, height, width).
        :param str | None attr_name: Name of the attribute to be added to the cells.
            If None, a random name will be generated. Default is None.
        :raises ValueError: If the shape of the data is not (1, height, width).
        """
        warnings.warn(
            "This method is deprecated. Use `add_property` instead.",
            stacklevel=2,
        )
        self.add_property(data, attr_name)

    def get_raster(self, attr_name: str | None = None) -> np.ndarray:
        """
        Return the values of given attribute.

        :param str | None attr_name: Name of the attribute to be returned. If None,
            returns all attributes. Default is None.
        :return: The values of given attribute as a 2D numpy array with shape (1, height, width).
        :rtype: np.ndarray
        """

        if attr_name is not None and attr_name not in self.attributes:
            raise ValueError(
                f"Attribute {attr_name} does not exist. "
                f"Choose from {self.attributes}, or set `attr_name` to `None` to retrieve all."
            )
        attr_names = self.attributes if attr_name is None else {attr_name}
        return np.stack([self.grid.property_layers[attr].data for attr in attr_names])

    def iter_neighborhood(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Coordinate]:
        """
        Return an iterator over cell coordinates that are in the
        neighborhood of a certain point.

        :param Coordinate pos: Coordinate tuple for the neighborhood to get.
        :param bool moore: Whether to use Moore neighborhood or not. If True,
            return Moore neighborhood (including diagonals). If False, return
            Von Neumann neighborhood (exclude diagonals).
        :param bool include_center: If True, return the (x, y) cell as well.
            Otherwise, return surrounding cells only. Default is False.
        :param int radius: Radius, in cells, of the neighborhood. Default is 1.
        :return: An iterator over cell coordinates that are in the neighborhood.
            For example with radius 1, it will return list with number of elements
            equals at most 9 (8) if Moore, 5 (4) if Von Neumann (if not including
            the center).
        :rtype: Iterator[Coordinate]
        """

        yield from self.get_neighborhood(pos, moore, include_center, radius)

    def iter_neighbors(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Cell]:
        """
        Return an iterator over neighbors to a certain point.

        :param Coordinate pos: Coordinate tuple for the neighborhood to get.
        :param bool moore: Whether to use Moore neighborhood or not. If True,
            return Moore neighborhood (including diagonals). If False, return
            Von Neumann neighborhood (exclude diagonals).
        :param bool include_center: If True, return the (x, y) cell as well.
            Otherwise, return surrounding cells only. Default is False.
        :param int radius: Radius, in cells, of the neighborhood. Default is 1.
        :return: An iterator of cells that are in the neighborhood; at most 9 (8)
            if Moore, 5 (4) if Von Neumann (if not including the center).
        :rtype: Iterator[Cell]
        """

        neighborhood = self.get_neighborhood(pos, moore, include_center, radius)
        return self.iter_cell_list_contents(neighborhood)

    @accept_tuple_argument
    def iter_cell_list_contents(
        self, cell_list: Iterable[Coordinate]
    ) -> Iterator[Cell]:
        """
        Returns an iterator of the contents of the cells
        identified in cell_list.

        :param Iterable[Coordinate] cell_list: Array-like of (x, y) tuples,
            or single tuple.
        :return: An iterator of the contents of the cells identified in cell_list.
        :rtype: Iterator[Cell]
        """

        # Note: filter(None, iterator) filters away an element of iterator that
        # is falsy. Hence, iter_cell_list_contents returns only non-empty
        # contents.
        return filter(None, (self.cells[x][y] for x, y in cell_list))

    @accept_tuple_argument
    def get_cell_list_contents(self, cell_list: Iterable[Coordinate]) -> list[Cell]:
        """
        Returns a list of the contents of the cells
        identified in cell_list.

        Note: this method returns a list of cells.

        :param Iterable[Coordinate] cell_list: Array-like of (x, y) tuples,
            or single tuple.
        :return: A list of the contents of the cells identified in cell_list.
        :rtype: List[Cell]
        """

        return list(self.iter_cell_list_contents(cell_list))

    def get_neighborhood(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> list[Coordinate]:
        """
        Get neighboring cell coordinates of a certain point.
        """
        cells = self.get_neighboring_cells(pos, moore, include_center, radius)
        return [cell.coordinate for cell in cells]

    def get_neighboring_cells(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> CellCollection:
        """
        Get neighboring cells of a certain point.
        """
        if moore != self._moore:
            raise NotImplementedError(
                "Only the same type of neighborhood is supported."
            )
        return self[pos[0], pos[1]].get_neighborhood(
            radius=radius, include_center=include_center
        )

    def to_crs(self, crs, inplace=False) -> RasterLayer | None:
        super()._to_crs_check(crs)
        layer = self if inplace else copy.copy(self)

        src_crs = rio.crs.CRS.from_user_input(layer.crs)
        dst_crs = rio.crs.CRS.from_user_input(crs)
        if not layer.crs.is_exact_same(crs):
            transform, dst_width, dst_height = calculate_default_transform(
                src_crs,
                dst_crs,
                self.width,
                self.height,
                *layer.total_bounds,
            )
            layer._total_bounds = [
                *transform_bounds(src_crs, dst_crs, *layer.total_bounds)
            ]
            layer.crs = crs
            layer._transform = transform

        if not inplace:
            return layer

    def to_image(self, colormap) -> ImageLayer:
        """
        Returns an ImageLayer colored by the provided colormap.
        """

        values = np.empty(shape=(4, self.height, self.width))
        for cell in self:
            row, col = cell.indices
            values[:, row, col] = colormap(cell)
        return ImageLayer(values=values, crs=self.crs, total_bounds=self.total_bounds)

    @classmethod
    def from_file(
        cls,
        raster_file: str,
        model: Model,
        cell_cls: type[Cell] = Cell,
        attr_name: str | None = None,
        rio_opener: Callable | None = None,
    ) -> RasterLayer:
        """
        Creates a RasterLayer from a raster file.

        :param str raster_file: Path to the raster file.
        :param Type[Cell] cell_cls: The class of the cells in the layer.
        :param str | None attr_name: The name of the attribute to use for the cell values.
            If None, a random name will be generated. Default is None.
        :param Callable | None rio_opener: A callable passed to Rasterio open() function.
        """

        with rio.open(raster_file, "r", opener=rio_opener) as dataset:
            values = dataset.read()
            _, height, width = values.shape
            total_bounds = [
                dataset.bounds.left,
                dataset.bounds.bottom,
                dataset.bounds.right,
                dataset.bounds.top,
            ]
            obj = cls(width, height, dataset.crs, total_bounds, model, cell_cls)
            obj._transform = dataset.transform
            obj.apply_raster(values, attr_name=attr_name)
            return obj

    def to_file(
        self, raster_file: str, attr_name: str | None = None, driver: str = "GTiff"
    ) -> None:
        """
        Writes a raster layer to a file.

        :param str raster_file: The path to the raster file to write to.
        :param str | None attr_name: The name of the attribute to write to the raster.
            If None, all attributes are written. Default is None.
        :param str driver: The GDAL driver to use for writing the raster file.
            Default is 'GTiff'. See GDAL docs at https://gdal.org/drivers/raster/index.html.
        """

        data = self.get_raster(attr_name)
        with rio.open(
            raster_file,
            "w",
            driver=driver,
            width=self.width,
            height=self.height,
            count=data.shape[0],
            dtype=data.dtype,
            crs=self.crs,
            transform=self.transform,
        ) as dataset:
            dataset.write(data)


class ImageLayer(RasterBase):
    _values: np.ndarray

    def __init__(self, values, crs, total_bounds):
        """
        Initializes an ImageLayer.

        :param values: The values of the image layer.
        :param crs: The coordinate reference system of the image layer.
        :param total_bounds: The bounds of the image layer in [min_x, min_y, max_x, max_y] format.
        """

        super().__init__(
            width=values.shape[2],
            height=values.shape[1],
            crs=crs,
            total_bounds=total_bounds,
        )
        self._values = values.copy()

    @property
    def values(self) -> np.ndarray:
        """
        Returns the values of the image layer.

        :return: The values of the image layer.
        :rtype: np.ndarray
        """

        return self._values

    @values.setter
    def values(self, values: np.ndarray) -> None:
        """
        Sets the values of the image layer.

        :param np.ndarray values: The values of the image layer.
        """

        self._values = values
        self._width = values.shape[2]
        self._height = values.shape[1]
        self._update_transform()

    def to_crs(self, crs, inplace=False) -> ImageLayer | None:
        super()._to_crs_check(crs)
        layer = self if inplace else copy.copy(self)

        src_crs = rio.crs.CRS.from_user_input(layer.crs)
        dst_crs = rio.crs.CRS.from_user_input(crs)
        if not layer.crs.is_exact_same(crs):
            num_bands, src_height, src_width = self.values.shape
            transform, dst_width, dst_height = calculate_default_transform(
                src_crs,
                dst_crs,
                src_width,
                src_height,
                *layer.total_bounds,
            )
            dst = np.empty(shape=(num_bands, dst_height, dst_width))
            for i, band in enumerate(layer.values):
                reproject(
                    source=band,
                    destination=dst[i],
                    src_transform=layer.transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )
            layer._total_bounds = [
                *transform_bounds(src_crs, dst_crs, *layer.total_bounds)
            ]
            layer._values = dst
            layer._height = layer._values.shape[1]
            layer._width = layer._values.shape[2]
            layer.crs = crs
            layer._transform = transform
        if not inplace:
            return layer

    @classmethod
    def from_file(cls, image_file) -> ImageLayer:
        """
        Creates an ImageLayer from an image file.

        :param image_file: The path to the image file.
        :return: The ImageLayer.
        :rtype: ImageLayer
        """

        with rio.open(image_file, "r") as dataset:
            values = dataset.read()
            total_bounds = [
                dataset.bounds.left,
                dataset.bounds.bottom,
                dataset.bounds.right,
                dataset.bounds.top,
            ]
            obj = cls(values=values, crs=dataset.crs, total_bounds=total_bounds)
            obj._transform = dataset.transform
            return obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(crs={self.crs}, total_bounds={self.total_bounds}, values={self.values!r})"
