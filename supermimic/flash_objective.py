"""Official BodyRecon objective plus the object keypoint reward, for FlashSAC training on the pristine flash_chord (PLAN.md 13.22.1).

Same term as Grace's modified flash_chord (ResMimic, arXiv 2510.05070): 6 points at +-radius along the object axes, Gaussian on their
mean squared distance to the reference points, so it covers rotation like the AUC and MPPE metrics. Grace weights it 3.0 in every
curriculum stage. An optional second, tighter Gaussian rewards the 1-3 cm accuracy the AUC thresholds start at. An optional finger term
rewards a closed hand on the frames that hand holds the object in the reference: the retargeted fingers never move, and the official
force closure pays only once a grasp already exists (ep 25 reaches the bottle with open fingers and knocks it away). An optional tail keeps paying force closure after
the last release: only the object is scored, and an object let go near the end can still tip over (ep 27 loses 31 cm in its last quarter). Training only:
`supermimic.kp_checkpoint` restores the official objective in the checkpoint config before official evaluation.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from flash_chord.objectives.recon_body import ReconBodyObjective, ReconBodyObjectiveConfig
from flash_chord.runtime.actions.sonic import SONIC_DIRECT_JOINT_NAMES


def fill_label_gaps(labels, wrist, obj_pos, obj_quat_xyzw, reach=0.08):
    """Contact labels (F, 2) without the gaps during which the hand stays on the same spot of the object: the retargeter's 2 cm test flickers (ep 1: 21 runs of its
    left hand, up to 63 frames off while it holds the handle), and force closure pays nothing on those frames."""
    rotation = Rotation.from_quat(obj_quat_xyzw)
    labels = labels.copy()
    for hand in range(2):
        rel = rotation.inv().apply(wrist[:, hand] - obj_pos)
        on = np.flatnonzero(np.diff(np.r_[0, (labels[:, hand] > 0.5).astype(int), 0]))
        for end, start in zip(on[1::2], on[2::2]):
            if np.linalg.norm(rel[end:start] - rel[max(end - 3, 0) : end].mean(0), axis=1).max() < reach:
                labels[end:start, hand] = 1.0
    return labels


def finger_grip_tables(objective, reference, config, reach=0.35, lift=0.15, ramp=15):
    """Closed finger pose per frame (F, fingers) and per-hand holding flag (F, 2), as core.finger_grip_table for the sides [left, right]."""
    names = list(SONIC_DIRECT_JOINT_NAMES)
    q_ids = list(objective.scalar_q_ids.numpy())
    slots = [q_ids.index(q) for q in objective.action.finger_q_ids.numpy()]
    lower, upper = objective.soft_lower.numpy()[slots], objective.soft_upper.numpy()[slots]
    closed = np.array([0.0 if "thumb_0" in n else config.finger_grip_frac * (lower[i] if abs(lower[i]) > abs(upper[i]) else upper[i])
                       for i, n in enumerate(names)], np.float32)
    obj = np.asarray(reference.object_body_pos_w(), np.float32)[:, 0]
    wrist = objective.reference.wrist_pos_w.numpy().reshape(len(obj), 2, 3)
    labels = objective.contact_active.numpy().reshape(len(obj), 2)
    near = np.linalg.norm(wrist - obj[:, None], axis=-1) < reach
    lifted = (obj[:, 2] > obj[0, 2] + lift)[:, None]
    holding = np.maximum(labels, lifted) * near
    closure = np.stack([np.convolve(holding[:, h], np.ones(ramp) / ramp, mode="same") for h in range(2)], 1)
    hand = np.array([0 if n.startswith("left") else 1 for n in names], np.int32)
    return closure[:, hand] * closed, (closure > 0).astype(np.float32), hand


@wp.kernel
def add_finger_grip(
    joint_q: wp.array(dtype=wp.float32),
    q_per_world: int,
    timestep: wp.array(dtype=wp.int32),
    num_frames: int,
    finger_q_ids: wp.array(dtype=wp.int32),
    finger_hand: wp.array(dtype=wp.int32),
    num_fingers: int,
    grip_q: wp.array(dtype=wp.float32),
    grip_on: wp.array(dtype=wp.float32),
    weight: float,
    std: float,
    frame_dt: float,
    score: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    frame = wp.clamp(timestep[world], 0, num_frames - 1)
    total = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    active = float(0.0)  # noqa: UP018
    for hand in range(2):
        if grip_on[frame * 2 + hand] > 0.5:
            sse = float(0.0)  # noqa: UP018
            count = float(0.0)  # noqa: UP018
            for finger in range(num_fingers):
                if finger_hand[finger] == hand:
                    error = joint_q[world * q_per_world + finger_q_ids[finger]] - grip_q[frame * num_fingers + finger]
                    sse += error * error
                    count += 1.0
            total += wp.exp(-(sse / count) / (std * std))
            active += 1.0
    if active > 0.0:
        score[world] += frame_dt * weight * total / active


@wp.kernel
def add_object_keypoints(
    body_q: wp.array(dtype=wp.transform),
    object_body_id: int,
    bodies_per_world: int,
    target_pos_w: wp.array(dtype=wp.vec3),
    target_quat_w: wp.array(dtype=wp.quat),
    object_radius: wp.array(dtype=wp.float32),
    weight: float,
    std: float,
    fine_weight: float,
    fine_std: float,
    frame_dt: float,
    score: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    pose = body_q[world * bodies_per_world + object_body_id]
    position = wp.transform_get_translation(pose)
    rotation = wp.transform_get_rotation(pose)
    sse = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    for keypoint in range(6):
        offset = wp.vec3(0.0, 0.0, 0.0)
        offset[keypoint % 3] = object_radius[world] * (1.0 - 2.0 * float(keypoint // 3))
        error = (position + wp.quat_rotate(rotation, offset)) - (target_pos_w[world] + wp.quat_rotate(target_quat_w[world], offset))
        sse += wp.dot(error, error)
    mean_squared = sse / 6.0
    total = weight * wp.exp(-mean_squared / (std * std)) + fine_weight * wp.exp(-mean_squared / (fine_std * fine_std))
    score[world] += frame_dt * total


@wp.kernel
def add_action_cost(
    action_rate_l2: wp.array(dtype=wp.float32),
    action_l2: wp.array(dtype=wp.float32),
    rate_weight: float,
    l2_weight: float,
    frame_dt: float,
    score: wp.array(dtype=wp.float32),
) -> None:
    world = wp.tid()
    score[world] -= frame_dt * (rate_weight * action_rate_l2[world] + l2_weight * action_l2[world])


@dataclass
class KeypointObjective(ReconBodyObjective):
    grip_q: wp.array | None = None
    grip_on: wp.array | None = None
    finger_hand: wp.array | None = None

    def evaluate(self, state, timestep: wp.array, terminated: wp.array) -> None:
        super().evaluate(state, timestep, terminated)
        if self.config.action_rate_extra_weight > 0.0 or self.config.action_l2_extra_weight > 0.0:
            wp.launch(
                add_action_cost,
                dim=self.world_count,
                inputs=[
                    self.action.action_rate_l2,
                    self.action.action_l2,
                    self.config.action_rate_extra_weight,
                    self.config.action_l2_extra_weight,
                    self.frame_dt,
                ],
                outputs=[self.score],
            )
        wp.launch(
            add_object_keypoints,
            dim=self.world_count,
            inputs=[
                state.body_q,
                self.object_body_id,
                self.bodies_per_world,
                self.command.body_target_pos_w,
                self.command.body_target_quat_w,
                self.command.object_radius,
                self.config.object_keypoints_weight,
                self.config.object_keypoints_std,
                self.config.object_keypoints_fine_weight,
                self.config.object_keypoints_fine_std,
                self.frame_dt,
            ],
            outputs=[self.score],
        )
        if self.grip_q is not None:
            wp.launch(
                add_finger_grip,
                dim=self.world_count,
                inputs=[
                    state.joint_q,
                    self.q_per_world,
                    timestep,
                    self.num_frames,
                    self.action.finger_q_ids,
                    self.finger_hand,
                    self.finger_hand.shape[0],
                    self.grip_q,
                    self.grip_on,
                    self.config.finger_grip_weight,
                    self.config.finger_grip_std,
                    self.frame_dt,
                ],
                outputs=[self.score],
            )


@dataclass(frozen=True)
class KeypointObjectiveConfig(ReconBodyObjectiveConfig):
    object_keypoints_weight: float = 3.0
    object_keypoints_std: float = 0.1
    object_keypoints_fine_weight: float = 0.0
    object_keypoints_fine_std: float = 0.03
    finger_grip_weight: float = 0.0
    finger_grip_std: float = 0.5
    finger_grip_frac: float = 0.6
    action_rate_extra_weight: float = 0.0  # on top of the official 1e-4; table A.2 of the SuperMimic paper is about 0.036 in these action units
    action_l2_extra_weight: float = 0.0  # on top of the official 1e-6: about 30 % of the actions of the kp policies sit at the tanh limit
    hold_tail: bool = False
    fill_label_gaps: bool = False

    def build(
        self,
        model,
        embodiment,
        robot_reference,
        reference,
        command,
        contact,
        action,
        world_count,
        frame_dt,
        device=None,
        reference_joint_q_offset=None,
        episode_start_frame=None,
    ):
        if reference_joint_q_offset is None or episode_start_frame is None:
            raise ValueError("ReconBody objective requires reset offset and episode-start buffers")
        objective = KeypointObjective.build(
            model,
            embodiment,
            robot_reference,
            reference,
            command,
            contact,
            action,
            reference_joint_q_offset,
            episode_start_frame,
            world_count,
            frame_dt,
            config=self,
            device=device,
        )
        if self.fill_label_gaps:
            frames = reference.object_body_pos_w().shape[0]
            filled = fill_label_gaps(
                objective.contact_active.numpy().reshape(frames, 2),
                objective.reference.wrist_pos_w.numpy().reshape(frames, 2, 3),
                np.asarray(reference.object_body_pos_w())[:, 0],
                np.asarray(reference.object_body_quat_w())[:, 0, [1, 2, 3, 0]],
            )
            objective.contact_active.assign(filled.reshape(-1))
        if self.hold_tail:
            labels = objective.contact_active.numpy().reshape(-1, 2).copy()
            for hand in range(2):
                held = np.flatnonzero(labels[:, hand] > 0.5)
                if len(held):
                    labels[held[-1] :, hand] = 1.0
            objective.contact_active.assign(labels.reshape(-1))
        if self.finger_grip_weight > 0.0:
            grip_q, grip_on, hand = finger_grip_tables(objective, reference, self)
            objective.grip_q = wp.array(grip_q.reshape(-1), dtype=wp.float32, device=device)
            objective.grip_on = wp.array(grip_on.reshape(-1), dtype=wp.float32, device=device)
            objective.finger_hand = wp.array(hand, dtype=wp.int32, device=device)
        return objective
