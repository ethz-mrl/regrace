from typing import Annotated, Literal

import numpy as np
import numpy.typing as npt
import pydantic


class PointCloud(pydantic.BaseModel):
    points: Annotated[npt.NDArray[np.float32], Literal[3, "N"]]
    pose: Annotated[npt.NDArray[np.float32],
                    Literal[4, 4]] = pydantic.Field(...,
                                                    min_length=4,
                                                    max_length=4)
    timestamp: float = pydantic.Field(..., ge=0)
    probability_labels: Annotated[npt.NDArray[np.float32], Literal["N", 20]]

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, points: Annotated[npt.NDArray[np.float32], Literal["N",
                                                                          3]],
                 pose: Annotated[npt.NDArray[np.float32],
                                 Literal[4, 4]], timestamp: float,
                 probability_labels: Annotated[npt.NDArray[np.float32],
                                               Literal["N", 20]]):
        # check input parameters
        assert points.shape[0] == probability_labels.shape[0], \
            f"number of points ({points.shape[0]}) and semantic_labels ({probability_labels.shape[0]}) do not match"
        assert probability_labels.shape[1] == 20, \
            "probability_labels are not of shape (N, 20)"
        assert points.shape[1] == 3, \
            "points are not of shape (N, 3)"
        assert pose.shape == (4, 4), \
            "pose is not of shape (4, 4)"
        assert np.allclose(sum(probability_labels.T), np.ones(points.shape[0])), \
            "probability_labels do not sum to 1"
        super().__init__(
            points=points.T,
            # transpose array and add ones to last column (homogeneous coordinates)
            probability_labels=probability_labels,
            timestamp=timestamp,
            pose=pose)

