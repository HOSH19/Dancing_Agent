import numpy as np
import mujoco

FOOT_GEOMS = ("right_foot", "left_foot")


def get_foot_contacts(model, data) -> dict:
    contacts = {}
    _f = np.zeros(6)
    for i in range(data.ncon):
        con = data.contact[i]
        g1 = model.geom(con.geom1).name
        g2 = model.geom(con.geom2).name
        for foot in FOOT_GEOMS:
            if foot in (g1, g2):
                mujoco.mj_contactForce(model, data, i, _f)
                contacts[foot] = contacts.get(foot, 0.0) + float(np.linalg.norm(_f[:3]))
    return contacts
