import pytest
import jax.numpy as jp
import numpy as np
from crazymarl.observations.multi_quad_observation import (
    parse_obs,
    get_ix4_mappings,
    get_obs_index_lookup,
)


def test_parse_obs_single_quad():
    num_quads = 1
    # define components
    payload_error = jp.array([0.1, 0.2, 0.3])        # (3,)
    payload_linvel = jp.array([0.4, 0.5, 0.6])       # (3,)
    rel_pos = jp.array([[1.0, 2.0, 3.0]])           # (1,3)
    rot_flat = jp.eye(3).reshape(-1)                # (9,)
    linvels = jp.array([[7.0, 8.0, 9.0]])           # (1,3)
    angvels = jp.array([[10.0, 11.0, 12.0]])        # (1,3)
    actions = jp.array([[0.1, -0.1, 0.2, -0.2]])    # (1,4)

    per_quad = jp.concatenate([
        rel_pos.reshape(-1),
        rot_flat,
        linvels.reshape(-1),
        angvels.reshape(-1),
        actions.reshape(-1),
    ])
    obs = jp.concatenate([payload_error, payload_linvel, per_quad], axis=0)

    pe, pl, rp, rots, lv, av, act = parse_obs(obs, num_quads)

    # shapes
    assert pe.shape == (3,)
    assert pl.shape == (3,)
    assert rp.shape == (1, 3)
    assert rots.shape == (1, 3, 3)
    assert lv.shape == (1, 3)
    assert av.shape == (1, 3)
    assert act.shape == (1, 4)

    # values
    assert np.allclose(pe, [0.1, 0.2, 0.3])
    assert np.allclose(pl, [0.4, 0.5, 0.6])
    assert np.allclose(rp, [[1.0, 2.0, 3.0]])
    assert np.allclose(rots[0], np.eye(3))
    assert np.allclose(lv, [[7.0, 8.0, 9.0]])
    assert np.allclose(av, [[10.0, 11.0, 12.0]])
    assert np.allclose(act, [[0.1, -0.1, 0.2, -0.2]])


@pytest.mark.parametrize("num_quads", [1, 2, 5])
def test_get_ix4_mappings_all_quads(num_quads):
    action_map, dyn_ix = get_ix4_mappings(num_quads)

    # keys
    expected_agents = {f"agent_{i}" for i in range(num_quads)}
    assert set(action_map.keys()) == expected_agents
    assert set(dyn_ix.keys()) == expected_agents

    # each action_map has 4 consecutive entries
    for i in range(num_quads):
        am = action_map[f"agent_{i}"]
        assert am.shape == (4,)
        assert jp.array_equal(am, jp.arange(4 * i, 4 * (i + 1), dtype=int))

    # each dyn_ix entry length: 3+3 payload +22 own +3*(num_quads-1) others
    expected_len = 6 + 22 + 3 * (num_quads - 1)
    for idxs in dyn_ix.values():
        assert idxs.ndim == 1
        assert idxs.shape[0] == expected_len


@pytest.mark.parametrize("num_quads", [1, 2, 5])
def test_obs_index_lookup_various(num_quads):
    global_ix, agent_ix = get_obs_index_lookup(num_quads)

    # 1) payload entries
    assert global_ix["payload_error"] == tuple(range(0, 3))
    assert global_ix["payload_linvel"] == tuple(range(3, 6))

    # 2) per-quad keys exist and have correct lengths
    for i in range(num_quads):
        base = 6 + i * 22
        # rel_pos
        assert len(global_ix[f"quad_{i}_rel_pos"]) == 3
        assert global_ix[f"quad_{i}_rel_pos"] == tuple(range(base, base + 3))
        # rot_flat
        assert len(global_ix[f"quad_{i}_rot_flat"]) == 9
        assert global_ix[f"quad_{i}_rot_flat"] == tuple(range(base + 3, base + 12))
        # linvel
        assert len(global_ix[f"quad_{i}_linvel"]) == 3
        assert global_ix[f"quad_{i}_linvel"] == tuple(range(base + 12, base + 15))
        # angvel
        assert len(global_ix[f"quad_{i}_angvel"]) == 3
        assert global_ix[f"quad_{i}_angvel"] == tuple(range(base + 15, base + 18))
        # action
        assert len(global_ix[f"quad_{i}_action"]) == 4
        assert global_ix[f"quad_{i}_action"] == tuple(range(base + 18, base + 22))

    # 3) aggregated groups lengths
    assert len(global_ix["all_rel_pos"]) == 3 * num_quads
    assert len(global_ix["all_rot_flat"]) == 9 * num_quads
    assert len(global_ix["all_linvel"]) == 3 * num_quads
    assert len(global_ix["all_angvel"]) == 3 * num_quads
    assert len(global_ix["all_action"]) == 4 * num_quads

    # 4) per-agent mapping matches dyn_ix & action_map
    action_map, dyn_map = get_ix4_mappings(num_quads)
    for i in range(num_quads):
        agent = f"agent_{i}"
        a_ix = agent_ix[agent]
        # shared payload
        assert a_ix["payload_error"] == global_ix["payload_error"]
        assert a_ix["payload_linvel"] == global_ix["payload_linvel"]
        # own rel_pos matches quad_{i}_rel_pos
        assert a_ix["own_rel_pos"] == global_ix[f"quad_{i}_rel_pos"]
        # others_rel_pos is concatenation of all other quad rel_pos
        other_idx = tuple(
            idx
            for j in range(num_quads) if j != i
            for idx in global_ix[f"quad_{j}_rel_pos"]
        )
        assert a_ix["others_rel_pos"] == other_idx
        # full_mapped_obs and action_map round-trip to get_ix4_mappings
        assert a_ix["full_mapped_obs"] == tuple(dyn_map[agent].tolist())
        assert a_ix["action_map"] == tuple(action_map[agent].tolist())