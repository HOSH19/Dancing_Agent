import numpy as np

FOOT_GEOMS = ("right_foot", "left_foot")


def get_foot_contacts(model, data) -> dict:
    contacts = {}
    for i in range(data.ncon):
        con = data.contact[i]
        g1 = model.geom(con.geom1).name
        g2 = model.geom(con.geom2).name
        for foot in FOOT_GEOMS:
            if foot in (g1, g2):
                contacts[foot] = contacts.get(foot, 0.0) + abs(np.linalg.norm(con.dist))
    return contacts
