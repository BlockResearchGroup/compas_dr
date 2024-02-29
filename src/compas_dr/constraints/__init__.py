"""This package defines geometric constraints that can be applied to the fixed vertices of a mesh/cable-mesh.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from compas.geometry import Line
from compas.geometry import Plane
from compas.geometry import Circle
from compas.geometry import NurbsCurve
from compas.geometry import NurbsSurface

from .constraint import Constraint

from .lineconstraint import LineConstraint
from .planeconstraint import PlaneConstraint
from .circleconstraint import CircleConstraint
from .curveconstraint import CurveConstraint
from .surfaceconstraint import SurfaceConstraint

Constraint.register(Line, LineConstraint)
Constraint.register(Plane, PlaneConstraint)
Constraint.register(Circle, CircleConstraint)
Constraint.register(NurbsCurve, CurveConstraint)
Constraint.register(NurbsSurface, SurfaceConstraint)
