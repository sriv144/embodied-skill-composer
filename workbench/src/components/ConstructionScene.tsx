import { Canvas, useThree } from "@react-three/fiber";
import { ContactShadows, Grid, Line, OrbitControls, RoundedBox, useGLTF } from "@react-three/drei";
import { Suspense, useEffect, useMemo } from "react";
import { Mesh, MeshStandardMaterial, Object3D } from "three";
import type { BuildModule, Project, ScheduledJob, TraceFrame, Vec3 } from "../types";

const materials: Record<string, string> = {
  foundation_concrete: "#555c5c",
  plaster_white: "#d8d9d2",
  interior_white: "#eeeae1",
  standing_seam_charcoal: "#252d2d"
};

type Props = {
  project: Project;
  frame?: TraceFrame;
  exploded?: boolean;
  selectedId?: string | null;
  onSelect?: (id: string) => void;
  routes?: ScheduledJob[];
};

export function ConstructionScene(props: Props) {
  return (
    <Canvas camera={{ position: [11, 9, -13], fov: 42 }} shadows dpr={[1, 1.75]} gl={{ antialias: true }}>
      <color attach="background" args={["#171c1b"]} />
      <fog attach="fog" args={["#171c1b", 22, 42]} />
      <ambientLight intensity={0.8} />
      <directionalLight position={[7, 14, -9]} intensity={3.2} castShadow shadow-mapSize={[2048, 2048]} />
      <Suspense fallback={<LoadingGeometry />}>
        <ResponsiveCamera />
        <Site {...props} />
      </Suspense>
      <Grid position={[0, -0.12, 0]} args={[28, 22]} cellSize={0.5} cellThickness={0.35} cellColor="#40504b" sectionSize={2} sectionThickness={0.7} sectionColor="#65766f" fadeDistance={28} infiniteGrid />
      <ContactShadows position={[0, -0.08, 0]} opacity={0.42} scale={28} blur={2.6} far={12} />
      <OrbitControls makeDefault target={[0, 1.2, 0]} minDistance={6} maxDistance={34} maxPolarAngle={Math.PI / 2.05} />
    </Canvas>
  );
}

function LoadingGeometry() {
  return <mesh position={[0, 0.02, 0]}><boxGeometry args={[0.2, 0.04, 0.2]} /><meshStandardMaterial color="#f2672e" /></mesh>;
}

function ResponsiveCamera() {
  const { camera, size } = useThree();
  useEffect(() => {
    const compact = size.width < 720;
    camera.position.set(compact ? 12.5 : 11, compact ? 10 : 9, compact ? -15.5 : -13);
    camera.lookAt(compact ? -0.5 : 0, 1.2, 0);
    if ("fov" in camera && typeof camera.fov === "number") {
      camera.fov = compact ? 52 : 42;
      camera.updateProjectionMatrix();
    }
  }, [camera, size.width]);
  return null;
}

function Site({ project, frame, exploded = false, selectedId, onSelect, routes = [] }: Props) {
  const stateByModule = useMemo(() => new Map(frame?.modules.map((item) => [item.module_id, item])), [frame]);
  const moduleStates = project.plan.modules.map((module, index) => {
    const state = stateByModule.get(module.module_id);
    const position = state?.position ?? module.target_pose.position;
    return {
      module,
      position: exploded ? explodedPosition(module.target_pose.position, index) : position,
      status: state?.status ?? "installed"
    };
  });
  const robots = frame?.robots ?? project.plan.robots.map((robot) => ({ robot_id: robot.robot_id, position: robot.start_pose.position, status: "idle", module_id: null }));
  return (
    <group>
      {project.geometry_asset_url ? (
        <NamedGLBModules assetUrl={project.geometry_asset_url} modules={moduleStates} selectedId={selectedId} onSelect={onSelect} />
      ) : moduleStates.map(({ module, position, status }) => (
        <FallbackModule key={module.module_id} module={module} position={position} selected={selectedId === module.module_id} state={status} onSelect={onSelect} />
      ))}
      {robots.map((robot, index) => (
        <RobotAsset
          key={robot.robot_id}
          position={formationPosition(robot, robots)}
          color={["#ff6b2c", "#eab83f", "#35a7a0", "#5d8ee6"][index]}
          assetUrl={project.robot_asset_url ?? `${import.meta.env.BASE_URL}demo/construction_robot.glb`}
        />
      ))}
      {frame && <RouteOverlay jobs={routes} timestamp={frame.timestamp_s} />}
      <mesh position={[-6.1, -0.09, 1.3]} receiveShadow><boxGeometry args={[5.2, 0.1, 9.2]} /><meshStandardMaterial color="#70452d" roughness={0.88} /></mesh>
    </group>
  );
}

function NamedGLBModules({
  assetUrl,
  modules,
  selectedId,
  onSelect
}: {
  assetUrl: string;
  modules: Array<{ module: BuildModule; position: Vec3; status: string }>;
  selectedId?: string | null;
  onSelect?: (id: string) => void;
}) {
  const { scene } = useGLTF(assetUrl);
  return <>{modules.map((item) => <GLBModule key={item.module.module_id} scene={scene} {...item} selected={selectedId === item.module.module_id} onSelect={onSelect} />)}</>;
}

function GLBModule({
  scene,
  module,
  position,
  status,
  selected,
  onSelect
}: {
  scene: Object3D;
  module: BuildModule;
  position: Vec3;
  status: string;
  selected: boolean;
  onSelect?: (id: string) => void;
}) {
  const object = useMemo(() => {
    const source = scene.getObjectByName(module.mesh_node);
    if (!source) return null;
    const clone = source.clone(true);
    clone.position.set(0, 0, 0);
    clone.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      child.castShadow = true;
      child.receiveShadow = true;
      if (Array.isArray(child.material)) child.material = child.material.map((item) => item.clone());
      else child.material = child.material.clone();
    });
    return clone;
  }, [module.mesh_node, scene]);
  useEffect(() => {
    object?.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      const list = Array.isArray(child.material) ? child.material : [child.material];
      list.forEach((material) => {
        material.transparent = status === "staged";
        material.opacity = status === "staged" ? 0.58 : 1;
        if (material instanceof MeshStandardMaterial) {
          material.emissive.set(selected ? "#f2672e" : "#000000");
          material.emissiveIntensity = selected ? 0.32 : 0;
        }
      });
    });
  }, [object, selected, status]);
  if (!object) return <FallbackModule module={module} position={position} state={status} selected={selected} onSelect={onSelect} />;
  return <primitive object={object} position={[position.x, position.z, -position.y]} onClick={(event: { stopPropagation: () => void }) => { event.stopPropagation(); onSelect?.(module.module_id); }} />;
}

function RobotAsset({ position, color, assetUrl }: { position: Vec3; color: string; assetUrl: string }) {
  const { scene } = useGLTF(assetUrl);
  const object = useMemo(() => {
    const clone = scene.clone(true);
    clone.traverse((child) => {
      if (!(child instanceof Mesh)) return;
      child.castShadow = true;
      child.receiveShadow = true;
      if (child.name.includes("upper_chassis")) {
        child.material = Array.isArray(child.material)
          ? child.material.map((material) => material.clone())
          : child.material.clone();
        const robotMaterials = Array.isArray(child.material) ? child.material : [child.material];
        robotMaterials.forEach((material) => {
          if (material instanceof MeshStandardMaterial) material.color.set(color);
        });
      }
    });
    return clone;
  }, [color, scene]);
  return (
    <group position={[position.x, 0, -position.y]}>
      <primitive object={object} scale={0.7} />
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.015, 0]}>
        <ringGeometry args={[0.54, 0.58, 40]} />
        <meshBasicMaterial color={color} transparent opacity={0.45} depthWrite={false} />
      </mesh>
    </group>
  );
}

function formationPosition(
  robot: { position: Vec3; module_id: string | null },
  robots: Array<{ position: Vec3; module_id: string | null }>
): Vec3 {
  if (!robot.module_id) return robot.position;
  const team = robots.filter((candidate) => candidate.module_id === robot.module_id);
  if (team.length < 2) return robot.position;
  const offset = (team.indexOf(robot) - (team.length - 1) / 2) * 0.72;
  return { ...robot.position, y: robot.position.y + offset };
}

function RouteOverlay({ jobs, timestamp }: { jobs: ScheduledJob[]; timestamp: number }) {
  const colors = ["#ff6b2c", "#eab83f", "#35a7a0", "#5d8ee6"];
  return <>{jobs.filter((job) => job.start_s <= timestamp && timestamp < job.end_s && job.route && job.route.length > 1).map((job, index) => (
    <Line
      key={job.module_id}
      points={job.route!.map((point) => [point.x, 0.055 + index * 0.008, -point.y] as [number, number, number])}
      color={colors[index % colors.length]}
      lineWidth={2}
      dashed
      dashSize={0.22}
      gapSize={0.12}
      transparent
      opacity={0.72}
    />
  ))}</>;
}

function FallbackModule({ module, position, selected, state, onSelect }: { module: BuildModule; position: Vec3; selected: boolean; state: string; onSelect?: (id: string) => void }) {
  const dimensions = module.dimensions;
  const rotation = module.target_pose.rotation_rpy_degrees;
  return (
    <group position={[position.x, Math.max(position.z, dimensions.height / 2), -position.y]} rotation={[(rotation.x * Math.PI) / 180, (-rotation.z * Math.PI) / 180, (-rotation.y * Math.PI) / 180]} onClick={(event) => { event.stopPropagation(); onSelect?.(module.module_id); }}>
      <RoundedBox args={[dimensions.width, dimensions.height, dimensions.depth]} radius={0.035} smoothness={2} castShadow receiveShadow>
        <meshStandardMaterial color={selected ? "#ff6b2c" : materials[module.material] ?? "#d8d9d2"} roughness={0.6} transparent opacity={state === "staged" ? 0.56 : 1} />
      </RoundedBox>
    </group>
  );
}

function explodedPosition(position: Vec3, index: number): Vec3 {
  const angle = index * 2.39996;
  const distance = 0.45 + index * 0.18;
  return { x: position.x + Math.cos(angle) * distance, y: position.y + Math.sin(angle) * distance, z: position.z + (index % 5) * 0.16 };
}
