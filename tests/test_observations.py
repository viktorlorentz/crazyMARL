import pytest
import jax.numpy as jp
import numpy as np
from crazymarl.observations.multi_quad_observation import parse_obs, get_ix4_mappings

def test_parse_obs_single_quad():
    num_quads = 1
    # define components
    payload_error = jp.array([0.1, 0.2, 0.3])        # (3,)
    payload_linvel = jp.array([0.4, 0.5, 0.6])       # (3,)
    rel_pos = jp.array([[1.0, 2.0, 3.0]])           # (1,3)
    # identity rotation matrix flattened
    rot_flat = jp.eye(3).reshape(-1)                # (9,)
    linvels = jp.array([[7.0, 8.0, 9.0]])           # (1,3)
    angvels = jp.array([[10.0, 11.0, 12.0]])        # (1,3)
    actions = jp.array([[0.1, -0.1, 0.2, -0.2]])    # (1,4)
    # build flat observation
    per_quad = jp.concatenate([rel_pos.reshape(-1), rot_flat, linvels.reshape(-1), angvels.reshape(-1), actions.reshape(-1)])
    obs = jp.concatenate([payload_error, payload_linvel, per_quad], axis=0)
    # parse
    pe, pl, rp, rots, lv, av, act = parse_obs(obs, num_quads)
    # assert shapes
    assert pe.shape == (3,)
    assert pl.shape == (3,)
    assert rp.shape == (1,3)
    assert rots.shape == (1,3,3)
    assert lv.shape == (1,3)
    assert av.shape == (1,3)
    assert act.shape == (1,4)
    # assert values
    assert np.allclose(pe, np.array([0.1,0.2,0.3]))
    assert np.allclose(pl, np.array([0.4,0.5,0.6]))
    assert np.allclose(rp, np.array([[1.0,2.0,3.0]]))
    assert np.allclose(rots[0], np.eye(3))
    assert np.allclose(lv, np.array([[7.0,8.0,9.0]]))
    assert np.allclose(av, np.array([[10.0,11.0,12.0]]))
    assert np.allclose(act, np.array([[0.1,-0.1,0.2,-0.2]]))


def test_get_ix4_mappings_two_quads():
    num_quads = 2
    action_map, dyn_ix = get_ix4_mappings(num_quads)
    # action_map keys and values
    assert set(action_map.keys()) == {"agent_0", "agent_1"}
    # each action_map entry has length 4 and correct indices
    assert jp.array_equal(action_map["agent_0"], jp.array([0,1,2,3]))
    assert jp.array_equal(action_map["agent_1"], jp.array([4,5,6,7]))
    # dyn_ix keys
    assert set(dyn_ix.keys()) == {"agent_0", "agent_1"}
    # expected length: 3 (payload_error) + 3 (payload_linvel) + 22 (own quad) + 3 (other rel_pos) = 31
    for agent, idxs in dyn_ix.items():
        assert idxs.ndim == 1
        assert idxs.shape[0] == 31
    # check specific entries for agent_0: first indices should be 0,1,2, then 3,4,5
    idx0 = dyn_ix["agent_0"]
    assert idx0[0] == 0 and idx0[1] == 1 and idx0[2] == 2
    assert idx0[3] == 3 and idx0[4] == 4 and idx0[5] == 5
    # own quad rel_pos start at index 6
    assert idx0[6] == 6
    # other quad rel_pos should include indices 6+22=28
    assert 28 in idx0
