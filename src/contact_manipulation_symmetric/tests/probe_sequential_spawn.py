"""Render and time the training spawn rule, one stage at a time.

Uses production USD assets, object-only ray observation and collision-checked
IK. No old role catalog, FFT packing, MILP, policy or physics integration is
used. Layout code is shared with training; geometry uses CPU, rays/IK CUDA.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'src/contact_manipulation_symmetric'),
             *(str(ROOT/'IsaacLab/source'/p) for p in ('isaaclab','isaaclab_tasks','isaaclab_rl'))]
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--num-layouts',type=int,default=1,help='Render consecutive seeds as separate layouts, reusing fixed assets.')
parser.add_argument('--margin',type=float,default=.01)
parser.add_argument('--timing-repeats',type=int,default=3)
parser.add_argument('--num-objects',type=int,default=None,help='Require exactly this many active objects in the diagnostic.')
parser.add_argument('--output',type=Path,default=ROOT/'docs/figures/sequential_spawn_2026-11-08')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
if args.num_layouts < 1:parser.error('num-layouts must be positive.')
if args.timing_repeats < 0 or not math.isfinite(args.margin) or args.margin < 0:
    parser.error('timing-repeats must be nonnegative and margin must be finite and nonnegative.')
if args.num_objects is not None:
    if args.num_objects < 4:parser.error('num-objects must be at least four.')
    # Only this diagnostic process changes its pool capacity, before importing
    # task configs. Training configuration files are left unchanged.
    os.environ['CONTACT_MANIPULATION_NUM_OBJECTS']=str(args.num_objects)
args.enable_cameras=True
launch_started=time.perf_counter()
app=AppLauncher(args).app

# Training and diagnostics execute exactly the same layout implementation.
import numpy as np
from contact_manipulation_symmetric.mdp.events import (
    SequentialSceneGenerator as SequentialSpawn,
    _sequential_corners as _corners,
    _sequential_rect as _rect,
    _sequential_rot as _rot,
)


def run():
    import numpy as np
    import torch
    import gymnasium as gym
    from PIL import Image,ImageDraw,ImageFont
    from pxr import Gf,Usd,UsdGeom,UsdPhysics,Vt
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils.math import quat_apply,quat_mul,convert_quat
    import contact_manipulation_symmetric
    from contact_manipulation_symmetric.config.ur5e.ur5e_env_cfg import UR5eContactManipulationEnvCfg
    from contact_manipulation_symmetric.mdp import events

    args.output.mkdir(parents=True,exist_ok=True)
    def synchronize():
        if args.device.startswith('cuda'):torch.cuda.synchronize()
    def timed(fn):
        synchronize();start=time.perf_counter();result=fn();synchronize()
        return result,(time.perf_counter()-start)*1000
    setup={}
    cfg=UR5eContactManipulationEnvCfg()
    cfg.seed=args.seed;cfg.scene.num_envs=1;cfg.sim.device=args.device
    if args.num_objects is not None:
        cfg.spawn_rules.min_objects=cfg.spawn_rules.max_objects=args.num_objects
    cfg.spawn_rules.all_asset_types=False
    # None of the existing scene-generation/reset events runs in this probe.
    cfg.events.reset_scene=None;cfg.events.reset_mdp=None
    cfg.spawn_rules.generation_mode='sequential_probe'
    cfg.scene.probe_camera=CameraCfg(prim_path='/World/SequentialProbeCamera',width=960,height=720,
        data_types=['rgb'],update_period=0.,spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.,horizontal_aperture=32.,clipping_range=(.01,30.)))
    def forbidden(*a,**kw):raise AssertionError('Old catalog/FFT/MILP must not be used in this probe.')
    events.initialize_certified_catalog=forbidden
    events._clutter_gpu_seed=forbidden
    events._solve_clutter_joint_domain=forbidden
    start=time.perf_counter()
    wrapped=gym.make('Isaac-Contact-Manipulation-Symmetric-UR5e-v0',cfg=cfg)
    setup['native_environment_ms']=(time.perf_counter()-start)*1000
    env=wrapped.unwrapped
    runtime,setup['geometry_mesh_bvh_robot_model_ms']=timed(lambda:events._initialize_episode_reset(env))
    assert not hasattr(runtime,'parameter_catalog')
    setup['launch_through_fixed_resources_ms']=(time.perf_counter()-launch_started)*1000
    rules=env.spawn_rules;geometries=env.spawn_geometries;scene=env.task_scene_data
    specs=runtime.specs;objects=env.scene['object_collection'];robot=env.scene['robot']
    resources=dict(rules=vars(rules),geometries=[dict(name=g.name,asset_name=g.asset_name,
        lower=list(map(float,g.lower)),upper=list(map(float,g.upper)),
        collision_footprint_vertices=[list(map(float,v)) for v in g.collision_footprint_vertices],
        visual_footprint_vertices=[list(map(float,v)) for v in g.visual_footprint_vertices]) for g in geometries])
    (args.output/'resources.json').write_text(json.dumps(resources,indent=2))
    camera=env.scene['probe_camera'];stage=sim_utils.get_current_stage()
    rgb_camera=UsdGeom.Camera(stage.GetPrimAtPath('/World/SequentialProbeCamera'))
    print('[Sequential] fixed resources ready; catalog/FFT/MILP calls guarded OFF',flush=True)
    ids=torch.zeros(1,dtype=torch.long,device=env.device)
    signs=np.array([(a,b,c) for a in (-.5,.5) for b in (-.5,.5) for c in (-.5,.5)])
    edges=[(i,j) for i in range(8) for j in range(i+1,8) if sum(signs[i]!=signs[j])==1]
    GREEN=(.05,1.,.22);RED=(1.,.08,.08);BLUE=(.15,.48,1.);YELLOW=(1.,.75,.02);CYAN=(.05,.95,1.);ORANGE=(1.,.35,.03);PINK=(1.,.12,.75)
    visual='/World/SequentialSpawnOverlay'
    UsdGeom.Xform.Define(stage,visual)
    xmin,xmax=rules.x_range;ymin,ymax=rules.y_range;top=rules.shelf_top
    cx,cy=(xmin+xmax)/2,(ymin+ymax)/2
    plane=UsdGeom.Mesh.Define(stage,visual+'/Support')
    plane.CreatePointsAttr([Gf.Vec3f(xmin,ymin,top-.002),Gf.Vec3f(xmax,ymin,top-.002),
        Gf.Vec3f(xmax,ymax,top-.002),Gf.Vec3f(xmin,ymax,top-.002)])
    plane.CreateFaceVertexCountsAttr([4]);plane.CreateFaceVertexIndicesAttr([0,1,2,3])
    plane.CreateDoubleSidedAttr(True);plane.CreateDisplayColorAttr([Gf.Vec3f(.25,.3,.35)])
    paths=[f'/World/envs/env_0/{s.name}' for s in specs]
    shelf_path='/World/envs/env_0/Shelf';robot_path='/World/envs/env_0/Robot'

    def visible(path,on):
        item=UsdGeom.Imageable(stage.GetPrimAtPath(path))
        item.MakeVisible() if on else item.MakeInvisible()
    line_id=0
    def clear():
        if stage.GetPrimAtPath(visual+'/Draw'):stage.RemovePrim(visual+'/Draw')
        UsdGeom.Xform.Define(stage,visual+'/Draw')
    def line(points,color,width=.0025):
        nonlocal line_id
        line_id+=1
        c=UsdGeom.BasisCurves.Define(stage,f'{visual}/Draw/L{line_id}')
        c.CreateTypeAttr('linear');c.CreateCurveVertexCountsAttr([len(points)])
        c.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float,p)) for p in points]))
        c.CreateWidthsAttr([width]);c.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    def rectangle(r,color,z=None,width=.003):
        a,b,c,d=r;z=top+.006 if z is None else z
        line([(a,c,z),(b,c,z),(b,d,z),(a,d,z),(a,c,z)],color,width)
    def obb(center,quat,extent,color,width=.002):
        local=torch.tensor(signs*np.array(extent),device=env.device,dtype=torch.float32)
        q=torch.tensor(quat,device=env.device,dtype=torch.float32).expand(8,-1)
        pts=quat_apply(q,local).cpu().numpy()+np.array(center)
        for i,j in edges:line([pts[i],pts[j]],color,width)
    def role_box(data,index,color):
        if index is None or index not in data['positions']:return
        g=geometries[index];p=data['positions'][index];yaw=data['yaws'][index]
        center=(*p[:2],p[2]+(g.lower[2]+g.upper[2])*.5)
        obb(center,(math.cos(yaw/2),0,0,math.sin(yaw/2)),np.asarray(g.upper)-g.lower,color)
    def overlay(data,ghosts=True):
        role_box(data,data.get('target_id'),GREEN);role_box(data,data.get('blocker_id'),RED)
        if data.get('clearance') is not None:rectangle(data['clearance'],BLUE)
        for r,goal in zip(data.get('pockets',[]),data.get('pocket_goals',[])):
            rectangle(r,YELLOW,width=.004)
            # Raise a second outline so the reserved empty area is visible in 3D.
            rectangle(r,YELLOW,z=top+.025,width=.0015)
            if ghosts:
                g=geometries[data['blocker_id']];x,y,z,yaw=goal
                # The rotated local box can extend outside the true rotated
                # footprint. Preview the same collision footprint used to
                # verify goal fit, rather than a larger box around the asset.
                polygon=events._collision_polygon(g,yaw)+np.array((x,y))
                for height in (top+.012,top+.045):
                    line([(px,py,height) for px,py in (*polygon,polygon[0])],YELLOW,.0012)
        fronts=[(corridor,ORANGE,top+.009) for corridor in data.get('pocket_front_corridors',[])]
        if data.get('blocker_front_corridor') is not None:
            fronts.append((data['blocker_front_corridor'],PINK,top+.014))
        for corridor,color,z in fronts:
            rectangle(corridor,color,z=z,width=.003)
            a,b,c,d=corridor
            if b-a>.025:
                start=(a+.005,(c+d)/2,z+.003);end=(b-.005,(c+d)/2,z+.003)
                line([start,end],color,.002)
                line([(end[0]-.012,end[1]-.007,end[2]),end,
                      (end[0]-.012,end[1]+.007,end[2])],color,.002)
    def make_sample(data):
        return events.SceneSample(tuple(data['active_ids']),data['target_id'],
            data.get('blocker_id') if data.get('blocker_id') is not None else -1,
            data['positions'],data['yaws'],data.get('clearance') or (xmin,xmax,ymin,ymax),
            tuple(data['pocket_goals'][0][:2]) if data.get('pocket_goals') else (cx,cy),
            clearance_kind='sequential_probe',reserved_regions=tuple(data.get('pockets',[])),
            camera_pose=data.get('camera_pose'),parameter_certificate={'method':'sequential_probe'})
    def tensors(data):
        sample=make_sample(data)
        if sample.active_ids:
            proposals=events._episode_proposals([sample],len(specs),env.device,scene.lower.dtype,
                max_objects=cfg.spawn_rules.max_objects)
        else:
            proposals=(scene.lower.new_zeros((1,len(specs),4)),
                torch.zeros((1,len(specs)),device=env.device,dtype=torch.bool),
                torch.full((1,cfg.spawn_rules.max_objects),-1,device=env.device,dtype=torch.long))
        states,active,slots=events.pack_episode_states(objects.data.default_object_state[0],[sample],
            scene.lower,scene.upper,scene.upright,proposals=proposals)
        return sample,proposals,states,active,slots
    def commit(data,show_robot=False):
        sample,proposed,states,active,slots=tensors(data)
        objects.set_active_objects(active,env_ids=ids)
        objects.write_object_state_to_sim(states)
        env.sim.forward()
        chosen=set(data['active_ids'])
        for i,p in enumerate(paths):visible(p,i in chosen)
        visible(robot_path,show_robot)
        return sample,proposed,states,active,slots
    def set_view(eye=None,aim=None,pose=None):
        rgb_camera.GetProjectionAttr().Set(UsdGeom.Tokens.perspective)
        rgb_camera.GetHorizontalApertureAttr().Set(32.)
        rgb_camera.GetVerticalApertureAttr().Set(24.)
        if pose is not None:
            camera.set_world_poses(torch.as_tensor(np.array([pose[:3]]),device=env.device,dtype=torch.float32),
                torch.as_tensor(np.array([pose[3:]]),device=env.device,dtype=torch.float32),convention='ros')
        else:
            camera.set_world_poses_from_view(torch.as_tensor(np.array([eye]),device=env.device,dtype=torch.float32),
                torch.as_tensor(np.array([aim]),device=env.device,dtype=torch.float32))
    def top_view():
        visible(shelf_path,False);visible(visual+'/Support',True)
        set_view(pose=(cx,cy,top+.88,0.,math.sqrt(.5),math.sqrt(.5),0.))
        # USD orthographic apertures are in tenths of stage length units.
        # Exact overhead rendering avoids the apparent overhang caused by
        # perspective magnification of tall objects in the previous figures.
        rgb_camera.GetProjectionAttr().Set(UsdGeom.Tokens.orthographic)
        rgb_camera.GetHorizontalApertureAttr().Set(10.*(ymax-ymin+.12))
        rgb_camera.GetVerticalApertureAttr().Set(10.*(ymax-ymin+.12)*.75)
    def perspective(whole=False):
        visible(shelf_path,True);visible(visual+'/Support',False)
        if whole:set_view((.8,-1.35,2.03),(-.35,0.,1.03))
        else:set_view((.35,-.87,1.90),(cx,cy,top+.09))
    def rgb():
        for _ in range(6):env.sim.render()
        camera.update(.001,force_recompute=True)
        return Image.fromarray(camera.data.output['rgb'][0,:,:,:3].cpu().numpy())
    font=ImageFont.truetype('DejaVuSans.ttf',26);small=ImageFont.truetype('DejaVuSans.ttf',18)
    frames=[]
    def save_pair(left,right,title,subtitle,filename):
        image=Image.new('RGB',(1920,820),(18,24,33));image.paste(left,(0,100));image.paste(right,(960,100))
        draw=ImageDraw.Draw(image);draw.text((20,10),title,font=font,fill='white')
        draw.text((20,53),subtitle,font=small,fill=(195,210,225))
        image.save(args.output/filename);frames.append(image)
    def selected_preview(data,index,yaw):
        preview=copy.deepcopy(data);preview['active_ids']=[index]
        g=geometries[index]
        preview['positions']={index:(cx,cy,top-g.lower[2])};preview['yaws']={index:yaw}
        preview['clearance']=None;preview['pockets']=[];preview['pocket_goals']=[]
        commit(preview);clear();role_box(preview,index,GREEN if index==data['target_id'] else RED)
        visible(shelf_path,False);visible(visual+'/Support',False)
        h=g.upper[2]-g.lower[2]
        set_view((cx+.35,cy-.35,top+h+.15),(cx,cy,top+h*.5))
        image=rgb();ImageDraw.Draw(image).text((24,24),'TYPE PREVIEW ONLY - NOT YET PLACED',font=small,fill='white')
        return image
    def render_stage(number,data,milliseconds,robot_start=None,vision=None):
        commit(data,show_robot=number==9);clear();overlay(data)
        descriptions={
            1:('Choose Target','Target type / yaw chosen; position is still unset.'),
            2:('Randomly place Target','Sample its feasible shelf domain, preserving room for Blocker and one or both side destinations.'),
            3:('Set Target-width clearance',f'BLUE A0 = Target frontal width plus {args.margin*1000:g} mm margin on each side.'),
            4:('Choose Blocker','Choose a non-Target type compatible with available space. RIGHT: type preview.'),
            5:('Place Blocker + reserve its front','10-50% Target frontal width overlap. PINK strip from full Blocker OBB front to shelf front stays empty.'),
            6:('Reserve adjoining destinations + empty fronts','YELLOW goals touch BLUE clearance. ORANGE front corridors stay empty; no full manipulation route certified.'),
            7:('Fill remaining space','Exclude Target, clearance, Blocker, destinations, destination fronts AND Blocker front.'),
            8:('Obtain object-only ray observations','LEFT diagnostic RGB; RIGHT selected observed OBBs. The policy uses 64 x 48 mesh rays.'),
            9:('Randomize robot start via batched IK','Elbow-up / limits / full robot collision checks and native articulation FK confirmation.'),
        }
        title,description=descriptions[number]
        if number==1:
            perspective();left=rgb()
            right=selected_preview(data,data['target_id'],data['metadata']['target_yaw'])
        elif number==4:
            top_view();left=rgb()
            right=selected_preview(data,data['blocker_id'],data['metadata']['blocker_yaw'])
        elif number==8:
            visible(shelf_path,True);visible(visual+'/Support',False);clear()
            set_view(pose=data['camera_pose']);left=rgb()
            root=robot.data.root_pose_w[0]
            for slot in vision.valid.nonzero().flatten().cpu().tolist():
                g=vision.geometry_b[slot]
                p=(root[:3]+quat_apply(root[3:],g[:3])).cpu().numpy()
                q=quat_mul(root[3:],g[3:7]).cpu().numpy()
                obb(p,q,g[7:].cpu().numpy(),GREEN if slot==0 else RED if slot==1 else CYAN,.0013)
            right=rgb()
        elif number==9:
            region=torch.as_tensor(robot_start[1]).detach().cpu().numpy()
            center=(region[::2]+region[1::2])*.5
            obb(center,(1.,0.,0.,0.),region[1::2]-region[::2],CYAN,.0012)
            perspective(True);left=rgb()
            b=np.array([*data['positions'][data['blocker_id']][:2],top+.15])
            set_view(b+np.array([.65,-.65,.4]),b+np.array([.05,0,.03]));right=rgb()
        else:
            top_view();left=rgb()
            perspective();right=rgb()
        save_pair(left,right,f'{number:02d}  {title}  |  generation {milliseconds:.3f} ms',description,
            f'{number:02d}_step.png')
        if number==7:
            for im,label,name in ((left,'Orthographic top view','final_top_view.png'),
                                  (right,'Perspective view','final_perspective.png')):
                standalone=Image.new('RGB',(960,820),(18,24,33));standalone.paste(im,(0,100))
                draw=ImageDraw.Draw(standalone)
                draw.text((20,12),f'{len(data["active_ids"])} objects | {label}',font=font,fill='white')
                draw.text((20,53),'BLUE clearance | YELLOW goals',font=small,fill=(195,210,225))
                draw.text((20,75),'PINK Blocker front | ORANGE destination fronts',font=small,fill=(195,210,225))
                standalone.save(args.output/name)

    def generate_layout(layout_seed,detailed):
        # All timings are synchronized wall times. PNGs and physics writes are
        # outside core generation timings; immutable resource setup is separate.
        records=[];snapshots=[]
        builder=SequentialSpawn(geometries,rules,layout_seed,args.margin)
        for number in range(1,8):
            names={1:'step1_target',2:'step2_place_target',3:'step3_clearance',4:'step4_blocker',
                   5:'step5_place_blocker',6:'step6_pockets',7:'step7_background'}
            data,elapsed=timed(getattr(builder,names[number]));snapshots.append(data)
            print(f'[Sequential] step{number}: {elapsed:.3f}ms, {len(data["active_ids"])} placed',flush=True)
            records.append(dict(step=number,core_ms=elapsed,
                                compute_device='CPU numpy geometry',snapshot=data))
        (args.output/'layout_stages.json').write_text(json.dumps(records,indent=2,default=float))
        final=snapshots[-1]
        if args.num_objects is not None and len(final['active_ids'])!=args.num_objects:
            raise RuntimeError(f'Exact-count diagnostic requires {args.num_objects}, got {len(final["active_ids"])}; no scene redraw.')
        for number,data,record in zip(range(1,8),snapshots,records):
            if detailed or number==7:
                _,render_ms=timed(lambda:render_stage(number,data,record['core_ms']))
            else:render_ms=0.
            record['diagnostic_commit_render_png_ms']=render_ms

        def observe(data):
            sample,proposed,states,active,slots=tensors(data)
            roots=robot.data.default_root_state[ids,:7].clone();roots[:,:3]+=env.scene.env_origins[ids]
            requests=[dict(env_id=0,active_ids=sample.active_ids,target_id=sample.target_id,blocker_id=sample.blocker_id)]
            snapshot=runtime.vision.observe_at_viewpoint(requests,
                torch.tensor([sample.camera_pose],device=env.device,dtype=states.dtype),
                object_poses=states[...,:7],root_poses=roots,retain_images=True)[0]
            return snapshot
        vision,elapsed=timed(lambda:observe(final))
        if detailed:_,render_ms=timed(lambda:render_stage(8,final,elapsed,vision=vision))
        else:render_ms=0.
        records.append(dict(step=8,core_ms=elapsed,diagnostic_commit_render_png_ms=render_ms,
                            compute_device=str(env.device),observed_ids=vision.observed_ids,valid=vision.valid,
                            geometry_b=vision.geometry_b,camera_pose_w=vision.camera_pose_w,intrinsics=vision.intrinsics))
        print(f'[Sequential] step8: {elapsed:.3f}ms, {int(vision.valid.sum())} observed slots',flush=True)

        def robot_start(data):
            sample,proposals,states,active,slots=tensors(data)
            anchor=events._sample_robot_starts(env,[sample],[layout_seed],proposals=proposals)[0]
            if anchor is None:
                replay,_=events._recover_initial_robot_starts(env,[sample],[layout_seed],[None],proposals)
                anchor=replay[0]
            if anchor is None:raise RuntimeError('New layout has no validated robot anchor; no scene redraw performed.')
            template=SimpleNamespace(joint_positions=torch.as_tensor(anchor[0],device=env.device,dtype=states.dtype)[None])
            cache=SimpleNamespace(sample=sample,proposals=proposals,template=template,robot=None)
            start=events._sample_robot_starts(env,[sample],[layout_seed+7919],scene_cache=cache)[0]
            used_anchor=False
            if start is None:
                start=events._sample_robot_starts(env,[sample],[layout_seed+15838],scene_cache=cache,recovery=True)[0]
            if start is None:start=anchor;used_anchor=True
            q=torch.as_tensor(start[0],device=env.device,dtype=states.dtype)[None]
            events._write_robot_reset(env,ids,q);env.sim.forward()
            poses=robot.data.body_link_pose_w[ids].clone();poses[...,3:]=convert_quat(poses[...,3:],to='xyzw')
            details={}
            valid=events._check_physical_robot_starts(env,[sample],[start],poses,env.scene.env_origins[ids],details=details)
            if not bool(valid.all()):raise RuntimeError(f'Physical articulation FK did not match: {details}')
            return start,dict(anchor_fallback=used_anchor,anchor_candidate=int(anchor[-1]),
                chosen_candidate=int(start[-1]),joint_names=robot.joint_names,joint_positions=q,
                region=start[1],palm_tcp=start[2],physical_fk_valid=True)
        synchronize();ik_started=time.perf_counter()
        try:
            start,start_report=robot_start(final)
        except RuntimeError as error:
            # Keep a failed IK diagnostic visible without redrawing the layout
            # or discarding the other preview scenes.
            start=None;start_report=dict(physical_fk_valid=False,error=str(error))
        synchronize();elapsed=(time.perf_counter()-ik_started)*1000
        if detailed and start is not None:
            _,render_ms=timed(lambda:render_stage(9,final,elapsed,robot_start=start))
        else:render_ms=0.
        records.append(dict(step=9,core_ms=elapsed,diagnostic_commit_render_png_ms=render_ms,
                            compute_device=str(env.device),robot_start=start_report))
        print(f'[Sequential] step9: {elapsed:.3f}ms, FK valid={start_report["physical_fk_valid"]}, '
              f'anchor fallback={start_report.get("anchor_fallback",False)}',flush=True)

        # Independent audit never retries or changes the generated scene.
        def validate(data):
            sample=make_sample(data);polygons={i:events._collision_polygon(geometries[i],data['yaws'][i])
                +np.asarray(data['positions'][i][:2]) for i in data['active_ids']}
            pairwise=[];outside=[];reserved=[]
            support=data['shelf_bounds']
            blocker_front=data['blocker_front_corridor']
            for slot,i in enumerate(sample.active_ids):
                p=polygons[i]
                if not (p[:,0].min()>=support[0]-1e-6 and p[:,0].max()<=support[1]+1e-6
                        and p[:,1].min()>=support[2]-1e-6 and p[:,1].max()<=support[3]+1e-6):outside.append(i)
                for j in sample.active_ids[slot+1:]:
                    if not events.polygons_separated(tuple(map(tuple,p)),tuple(map(tuple,polygons[j])),0.):pairwise.append((i,j))
                if i not in (sample.target_id,sample.blocker_id):
                    for r in (data['clearance'],*data['pockets'],*data.get('pocket_front_corridors',[]),blocker_front):
                        if events.overlap_area(tuple(map(tuple,p)),events.rectangle_polygon(r))>1e-10:reserved.append(i)
            counts={}
            for i in sample.active_ids:counts[specs[i].asset_name]=counts.get(specs[i].asset_name,0)+1
            pocket_checks=[]
            for pocket_slot,(r,goal) in enumerate(zip(data['pockets'],data['pocket_goals'])):
                gp=events._collision_polygon(geometries[sample.blocker_id],goal[3])+goal[:2]
                g=geometries[sample.blocker_id]
                half=(np.asarray(g.upper[:2])-g.lower[:2])*.5
                goal_obb=_rot(_corners((-half[0],half[0],-half[1],half[1])),goal[3])+goal[:2]
                obb_padding=min(goal_obb[:,0].min()-r[0],r[1]-goal_obb[:,0].max(),
                                goal_obb[:,1].min()-r[2],r[3]-goal_obb[:,1].max())
                obb_clear=events.overlap_area(tuple(map(tuple,goal_obb)),events.rectangle_polygon(data['clearance']))<=1e-10
                inside=(gp[:,0].min()>=r[0]-1e-6 and gp[:,0].max()<=r[1]+1e-6
                        and gp[:,1].min()>=r[2]-1e-6 and gp[:,1].max()<=r[3]+1e-6)
                goal_outside=events.overlap_area(tuple(map(tuple,gp)),events.rectangle_polygon(data['clearance']))<=1e-10
                region_outside=events.overlap_area(events.rectangle_polygon(r),events.rectangle_polygon(data['clearance']))<=1e-10
                empty=all(events.overlap_area(tuple(map(tuple,p)),events.rectangle_polygon(r))<=1e-10 for p in polygons.values())
                aside=r[3]<=data['clearance'][2]+1e-6 or r[2]>=data['clearance'][3]-1e-6
                adjoining=(abs(r[3]-data['clearance'][2])<=1e-8 or abs(r[2]-data['clearance'][3])<=1e-8)
                corridor=data['pocket_front_corridors'][pocket_slot]
                front_empty=all(events.overlap_area(tuple(map(tuple,p)),events.rectangle_polygon(corridor))<=1e-10
                                for p in polygons.values())
                pocket_checks.append(dict(blocker_fits=bool(inside),empty=bool(empty),outside_clearance=bool(aside),
                    rotated_obb_outside_clearance=bool(obb_clear),obb_padding_sufficient=bool(obb_padding>=rules.gap-1e-6),
                    rotated_goal_outside_clearance=bool(goal_outside),entire_region_outside_clearance=bool(region_outside),
                    adjoining_clearance=bool(adjoining),front_corridor_empty=bool(front_empty)))
            target_once=counts[specs[sample.target_id].asset_name]==1
            non_target_counts=[count for name,count in counts.items() if name!=specs[sample.target_id].asset_name]
            count_range=(int(rules.min_objects)<=len(sample.active_ids)<=int(rules.max_objects))
            balanced=max(non_target_counts)-min(non_target_counts)<=1
            sides=[-1 if r[3]<=data['clearance'][2]+1e-8 else
                    1 if r[2]>=data['clearance'][3]-1e-8 else 0 for r in data['pockets']]
            valid_sides=(1<=len(sides)<=2 and 0 not in sides and len(set(sides))==len(sides))
            blocker_front_empty=all(events.overlap_area(tuple(map(tuple,p)),events.rectangle_polygon(blocker_front))<=1e-10
                                    for p in polygons.values())
            valid=(not pairwise and not outside and not reserved and target_once and valid_sides
                and count_range and balanced and blocker_front_empty
                and all(all(x.values()) for x in pocket_checks)
                and .1-1e-8<=data['metadata']['frontal_overlap_ratio']<=.5+1e-8)
            result=dict(valid=valid,overlapping_object_pairs=pairwise,objects_outside_support=outside,
                background_in_reserved_regions=reserved,pockets=pocket_checks,target_type_unique=target_once,
                at_least_one_destination=bool(valid_sides),destination_sides=sides,
                blocker_front_corridor_empty=bool(blocker_front_empty),
                count_in_requested_range=bool(count_range),non_target_type_counts_balanced=bool(balanced),
                type_counts=counts,frontal_overlap_ratio=data['metadata']['frontal_overlap_ratio'])
            if not valid:raise AssertionError(result)
            return result
        audit,audit_ms=timed(lambda:validate(final))

        # Same seed, identical geometric inputs, no new layouts: separate warmed
        # kernel/cache timing from the single illustrated cold execution.
        warm=[]
        for repeat in range(args.timing_repeats):
            b=SequentialSpawn(geometries,rules,layout_seed,args.margin);row=[]
            for name in ('step1_target','step2_place_target','step3_clearance','step4_blocker',
                         'step5_place_blocker','step6_pockets','step7_background'):
                d,ms=timed(getattr(b,name));row.append(ms)
            assert d==final,'Same-seed repetition must not change the illustrated layout.'
            _,ms=timed(lambda:observe(d));row.append(ms)
            _,ms=timed(lambda:robot_start(d));row.append(ms)
            warm.append(row)
        for prim in Usd.PrimRange(stage.GetPrimAtPath(visual)):
            assert not prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.HasAPI(UsdPhysics.CollisionAPI)
        # Persist the measured geometries so the construction is independently
        # reproducible on CPU without loading Isaac Sim or a parameter catalog.
        def jsonable(v):
            if isinstance(v,torch.Tensor):return v.detach().cpu().tolist()
            if isinstance(v,np.ndarray):return v.tolist()
            if isinstance(v,np.generic):return v.item()
            if isinstance(v,dict):return {str(k):jsonable(x) for k,x in v.items()}
            if isinstance(v,(tuple,list)):return [jsonable(x) for x in v]
            if hasattr(v,'__dataclass_fields__'):return {k:jsonable(getattr(v,k)) for k in v.__dataclass_fields__}
            return v
        report=dict(seed=layout_seed,margin_per_side_m=args.margin,
            diagnostic_composition=dict(exact_count=args.num_objects),setup=setup,steps=records,
            warm_repeats_ms=warm,warm_median_ms=np.median(warm,axis=0) if warm else None,
            total_core_cold_ms=sum(r['core_ms'] for r in records),audit=audit,audit_ms=audit_ms,
            final=final,geometry_resource_file='../resources.json' if args.num_layouts>1 else 'resources.json',spawn_rules=rules,
            physical_steps_after_initial_resource_setup=0,production_reset_changed=False,
            limitations=['One native environment reused across layouts; not a 1024-environment benchmark.',
                'CPU numpy layout math; CUDA production raycast / batched IK.',
                'Conservative empty rectangles; continuous background yaw conditioned by fitting domains.',
                'Target placement conservatively reserves room for initial Blocker and side goals.',
                'Goal yaw is sampled independently of initial Blocker yaw, conditioned on available space.',
                'Exact requested count is checked for this example; no all-seed capacity guarantee.',
                'Destinations adjoin clearance and their shelf-front corridors are empty; no complete pushing/pivot path certificate.',
                'Exactly bounded width overlap, not silhouette-area visibility.',
                'Margin 10 mm/side is provisional for this prototype.'])
        (args.output/'report.json').write_text(json.dumps(jsonable(report),indent=2,ensure_ascii=False))
        with (args.output/'timings.csv').open('w') as f:
            f.write('step,first_call_core_ms,warm_median_core_ms,diagnostic_commit_render_png_ms\n')
            for i,r in enumerate(records):
                f.write(f'{i+1},{r["core_ms"]:.6f},{np.median(warm,axis=0)[i] if warm else float("nan"):.6f},{r["diagnostic_commit_render_png_ms"]:.6f}\n')
        montage=Image.new('RGB',(1920,273*math.ceil(len(frames)/3)),(18,24,33))
        for i,image in enumerate(frames):
            thumb=image.resize((640,273));montage.paste(thumb,((i%3)*640,(i//3)*273))
        montage.save(args.output/'overview.png')
        print('[Sequential] COMPLETE '+json.dumps(audit),flush=True)
        return report

    base_output=args.output
    reports=[]
    for layout_index in range(args.num_layouts):
        args.output=base_output/f'layout_{layout_index+1:02d}' if args.num_layouts>1 else base_output
        args.output.mkdir(parents=True,exist_ok=True)
        frames.clear()
        print(f'[Sequential] LAYOUT {layout_index+1}/{args.num_layouts}, seed {args.seed+layout_index}',flush=True)
        reports.append(generate_layout(args.seed+layout_index,detailed=layout_index==0))
    args.output=base_output
    if args.num_layouts>1:
        columns=5;thumb_width=384
        for source,target in (('final_top_view.png','all_top_views.png'),('final_perspective.png','all_perspective_views.png')):
            is_top=source=='final_top_view.png'
            image_height=184 if is_top else 288
            thumb_height=image_height+30
            sheet=Image.new('RGB',(columns*thumb_width,math.ceil(args.num_layouts/columns)*thumb_height),(18,24,33))
            for i,report in enumerate(reports):
                path=base_output/f'layout_{i+1:02d}'/source
                raw=Image.open(path)
                crop=raw.crop((30,245,930,675) if is_top else (0,100,960,820))
                tile=Image.new('RGB',(thumb_width,thumb_height),(18,24,33))
                tile.paste(crop.resize((thumb_width,image_height)),(0,30))
                ImageDraw.Draw(tile).text((8,4),f'{i+1:02d} | {len(report["final"]["active_ids"])} objects | seed {report["seed"]}',font=small,fill='white')
                sheet.paste(tile,((i%columns)*thumb_width,(i//columns)*thumb_height))
            sheet.save(base_output/target)
        summary=[]
        with (base_output/'timings.csv').open('w') as f:
            f.write('layout,seed,objects,step1_ms,step2_ms,step3_ms,step4_ms,step5_ms,step6_ms,step7_ms,raycast_ms,ik_ms,total_core_ms,render_commit_png_ms,destination_sides,target_blocker_depth_gap_m\n')
            for i,r in enumerate(reports):
                row=dict(layout=i+1,seed=r['seed'],objects=len(r['final']['active_ids']),
                    step_ms=[x['core_ms'] for x in r['steps']],total_core_ms=r['total_core_cold_ms'],
                    render_commit_png_ms=sum(x['diagnostic_commit_render_png_ms'] for x in r['steps']),
                    target_blocker_depth_gap_m=r['final']['metadata']['target_blocker_depth_gap_m'],
                    audit=r['audit'],vision_valid=r['steps'][7]['valid'],robot_start=r['steps'][8]['robot_start'])
                summary.append(row)
                side_label=';'.join('left' if side<0 else 'right' for side in row['audit']['destination_sides'])
                values=[i+1,row['seed'],row['objects'],*row['step_ms'],row['total_core_ms'],row['render_commit_png_ms'],side_label,row['target_blocker_depth_gap_m']]
                f.write(','.join(str(x) for x in values)+'\n')
        steps=np.asarray([r['step_ms'] for r in summary])
        metrics=dict(layout_steps_1_7_ms=steps[:,:7].sum(axis=1),raycast_ms=steps[:,7],
            robot_search_write_fk_ms=steps[:,8],core_steps_1_9_ms=steps.sum(axis=1),
            render_commit_png_ms=np.asarray([r['render_commit_png_ms'] for r in summary]))
        statistics={key:dict(mean=float(values.mean()),median=float(np.median(values)),
            p95=float(np.percentile(values,95)),min=float(values.min()),max=float(values.max()),
            sum=float(values.sum())) for key,values in metrics.items()}
        statistics['per_step_mean_ms']=steps.mean(axis=0).tolist()
        statistics['per_step_median_ms']=np.median(steps,axis=0).tolist()
        (base_output/'summary.json').write_text(json.dumps(dict(setup=setup,layouts=summary,statistics=statistics),indent=2,default=lambda x:x.detach().cpu().tolist() if isinstance(x,torch.Tensor) else float(x)))
        print(f'[Sequential] BATCH COMPLETE {len(reports)} layouts',flush=True)
    wrapped.close()

if __name__=='__main__':
    try:run()
    except Exception:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'error.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        raise
    finally:app.close(wait_for_replicator=False,skip_cleanup=True)
