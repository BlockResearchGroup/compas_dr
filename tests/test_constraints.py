import pytest
from compas.geometry import NurbsSurface
from compas.geometry import Point
from compas.geometry import Vector

from compas_dr.constraints import LineConstraint
from compas_dr.constraints import SurfaceConstraint


def test_line_constraint_loads_compas2_json():
    serialized = (
        '{"dtype":"compas_dr.constraints/LineConstraint",'
        '"data":{"geometry":{"start":[0.0,0.0,0.0],"end":[1.0,0.0,0.0]}},'
        '"guid":"51e745e5-611a-4377-8f39-0125373d55de","name":"support"}'
    )
    constraint = LineConstraint.from_jsonstring(serialized)

    assert isinstance(constraint, LineConstraint)
    assert constraint.__data__ == {"geometry": {"start": [0.0, 0.0, 0.0], "end": [1.0, 0.0, 0.0]}}
    assert constraint.name == "support"
    assert str(constraint.guid) == "51e745e5-611a-4377-8f39-0125373d55de"
    assert LineConstraint.from_jsonstring(constraint.to_jsonstring()).__data__ == constraint.__data__


def test_nurbs_surface_constraint_retains_legacy_payload_and_projects(monkeypatch: pytest.MonkeyPatch) -> None:
    from compas.geometry.surfaces import nurbs

    control_points = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]

    class TestSurface(NurbsSurface):
        @property
        def __data__(self) -> dict[str, object]:
            return {"control_points": control_points}

    def load_surface(cls: type[NurbsSurface], data: dict[str, object]) -> NurbsSurface:
        assert data == {"control_points": control_points}
        return TestSurface()

    def closest_point(_surface: NurbsSurface, _point: Point, *, return_parameters: bool) -> Point | tuple[Point, tuple[float, float]]:
        point = Point(0.5, 0.5, 0.0)
        return (point, (0.5, 0.5)) if return_parameters else point

    def normal_at(_surface: NurbsSurface, _u: float, _v: float) -> Vector:
        return Vector(0.0, 0.0, 1.0)

    monkeypatch.setattr(NurbsSurface, "__from_data__", classmethod(load_surface))
    monkeypatch.setattr(nurbs, "surface_closest_point", closest_point)
    monkeypatch.setattr(nurbs, "surface_normal_at", normal_at)
    legacy = {"geometry": {"points": control_points}, "rhino_guid": "support"}
    constraint = SurfaceConstraint.__from_data__(legacy)
    constraint.location = [0.5, 0.5, 1.0]
    constraint.residual = [1.0, 2.0, 3.0]

    assert constraint.__data__ == legacy
    assert legacy["geometry"] == {"points": control_points}
    assert constraint.location == Point(0.5, 0.5, 0.0)
    assert constraint.tangent == Vector(1.0, 2.0, 0.0)
    assert constraint.normal == Vector(0.0, 0.0, 3.0)
