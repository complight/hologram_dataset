import bpy
import sys
import os
import numpy as np

cur_dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(cur_dir)

import libblend

def main():
    libblend.prepare()
    directory = '{}/output'.format(cur_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(0,100):
        fn = '{}/image_{:04d}.png'.format(directory,i) 
        render(fn)
    return True

def render(fn,copies=12):
    inventory = []
    inventory.append(bpy.context.scene.objects['cone_dummy'])
    inventory.append(bpy.context.scene.objects['cube_dummy'])
    inventory.append(bpy.context.scene.objects['ball_dummy'])
    c         = bpy.context
    new_items = []
    zs        = np.linspace(-5,15,4)
    z         = np.random.choice(zs,copies)
    teta      = np.random.uniform(-30,30,copies)
    beta      = np.random.uniform(-20,20,copies)
    teta      = np.radians(teta)
    beta      = np.radians(beta)
    inv_ids   = np.random.randint(len(inventory),size=copies)
    for i in range(copies):
        inv_id = inv_ids[i]
        new_items.append(inventory[inv_id].copy())
        new_items[-1].data        = inventory[inv_id].data.copy()
        new_items[-1].hide_render = False
        new_items[-1].location    = (
                                     float((z[i]+15)*np.sin(teta[i])),
                                     float((z[i]+15)*np.sin(beta[i])),
                                     float((z[i]+15))
                                    )
        c.collection.objects.link(new_items[-1])
    libblend.set_render_type(render_type='depth')
    libblend.render(fn.replace('.png','_depth.png'))
    libblend.set_render_type(render_type='rgb')
    libblend.render(fn)
    for i in range(copies):
        bpy.data.objects.remove(new_items[i],do_unlink=True)
    return True

main()
