import { Canvas, useThree } from "@react-three/fiber";
import { ContactShadows, Environment, Grid, OrbitControls, RoundedBox } from "@react-three/drei";
import { Suspense, useEffect, useMemo } from "react";
import type { BuildModule, Project, TraceFrame, Vec3 } from "../types";

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
};

export function ConstructionScene(props: Props) {
  return (
    <Canvas camera={{ position: [11, 9, -13], fov: 42 }} shadows dpr={[1, 1.75]}>
      <color attach="background" args={["#171c1b"]} />
      <fog attach="fog" args={["#171c1b", 22, 42]} />
      <ambientLight intensity={0.8} />
      <directionalLight position={[7, 14, -9]} intensity={3.2} castShadow shadow-mapSize={[2048, 2048]} />
      <Suspense fallback={null}>
        <ResponsiveCamera />
        <Site {...props} />
        <Environment preset="warehouse" environmentIntensity={0.45} />
      </Suspense>
      <Grid
        position={[0, -0.12, 0]}
        args={[28, 22]}
        cellSize={0.5}
        cellThickness={0.35}
        cellColor="#40504b"
        sectionSize={2}
        sectionThickness={0.7}
        sectionColor="#65766f"
        fadeDistance={28}
        infiniteGrid
      />
      <ContactShadows position={[0, -0.08, 0]} opacity={0.42} scale={28} blur={2.6} far={12} />
      <OrbitControls makeDefault target={[0, 1.2, 0]} minDistance={6} maxDistance={34} maxPolarAngle={Math.PI / 2.05} />
    </Canvas>
  );
}

function ResponsiveCamera() {
  const { camera, size } = useThree();
  useEffect(() => {
    if (size.width < 720) {
      camera.position.set(14, 11, -17);
    } else {
      camera.position.set(11, 9, -13);
    }
    camera.lookAt(0, 1.2, 0);
    if ("fov" in camera && typeof camera.fov === "number") {
      camera.fov = size.width < 720 ? 50 : 42;
      camera.updateProjectionMatrix();
    }
  }, [camera, size.width]);
  return null;
}

function Site({ project, frame, exploded = false, selectedId, onSelect }: Props) {
  const stateByModule = useMemo(
    () => new Map(frame?.modules.map((item) => [item.module_id, item])),
    [frame]
  );
  return (
    <group>
      {project.plan.modules.map((module, index) => {
        const state = stateByModule.get(module.module_id);
        const position = state?.position ?? module.target_pose.position;
        const display = exploded ? explodedPosition(module.target_pose.position, index) : position;
        return (
          <ModuleMesh
            key={module.module_id}
            module={module}
            position={display}
            selected={selectedId === module.module_id}
            state={state?.status ?? "installed"}
            onSelect={onSelect}
          />
        );
      })}
      {(frame?.robots ?? project.plan.robots.map((robot) => ({
        robot_id: robot.robot_id,
        position: robot.start_pose.position,
        status: "idle",
        module_id: null
      }))).map((robot, index) => (
        <Robot key={robot.robot_id} position={robot.position} color={["#ff6b2c", "#eab83f", "#35a7a0", "#5d8ee6"][index]} />
      ))}
      <mesh position={[-6.1, -0.09, -1.3]} receiveShadow>
        <boxGeometry args={[5.2, 0.1, 9.2]} />
        <meshStandardMaterial color="#70452d" roughness={0.88} />
      </mesh>
    </group>
  );
}

function ModuleMesh({
  module,
  position,
  selected,
  state,
  onSelect
}: {
  module: BuildModule;
  position: Vec3;
  selected: boolean;
  state: string;
  onSelect?: (id: string) => void;
}) {
  const d = module.dimensions;
  const rotation = module.target_pose.rotation_rpy_degrees;
  const opacity = state === "staged" ? 0.56 : 1;
  const color = selected ? "#ff6b2c" : materials[module.material] ?? "#d8d9d2";
  return (
    <group
      position={[position.x, Math.max(position.z, d.height / 2), position.y]}
      rotation={[
        (rotation.x * Math.PI) / 180,
        (-rotation.z * Math.PI) / 180,
        (-rotation.y * Math.PI) / 180
      ]}
      onClick={(event) => {
        event.stopPropagation();
        onSelect?.(module.module_id);
      }}
    >
      <RoundedBox args={[d.width, d.height, d.depth]} radius={0.035} smoothness={2} castShadow receiveShadow>
        <meshStandardMaterial color={color} roughness={0.6} transparent opacity={opacity} />
      </RoundedBox>
      {module.module_type === "door_panel" && (
        <mesh position={[0, -d.height * 0.13, d.depth / 2 + 0.012]}>
          <boxGeometry args={[Math.min(1, d.width * 0.56), d.height * 0.72, 0.04]} />
          <meshStandardMaterial color="#714022" roughness={0.7} />
        </mesh>
      )}
      {module.module_type === "window_panel" && (
        <mesh position={[0, 0, d.depth / 2 + 0.014]}>
          <boxGeometry args={[Math.min(1.2, d.width * 0.58), d.height * 0.46, 0.045]} />
          <meshPhysicalMaterial color="#247c87" roughness={0.16} transmission={0.35} />
        </mesh>
      )}
      {module.module_type === "roof_panel" && [0, 1, 2, 3, 4].map((rib) => (
        <mesh key={rib} position={[-d.width / 2 + (d.width * rib) / 4, d.height / 2 + 0.025, 0]}>
          <boxGeometry args={[0.025, 0.05, d.depth * 0.98]} />
          <meshStandardMaterial color="#d76632" />
        </mesh>
      ))}
    </group>
  );
}

function Robot({ position, color }: { position: Vec3; color: string }) {
  return (
    <group position={[position.x, 0.2, position.y]}>
      <RoundedBox args={[0.68, 0.22, 0.48]} radius={0.06} smoothness={3} castShadow>
        <meshStandardMaterial color={color} metalness={0.15} roughness={0.4} />
      </RoundedBox>
      {[-1, 1].flatMap((x) => [-1, 1].map((y) => (
        <mesh key={`${x}-${y}`} position={[x * 0.27, -0.08, y * 0.27]} rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[0.105, 0.105, 0.09, 18]} />
          <meshStandardMaterial color="#111514" roughness={0.8} />
        </mesh>
      )))}
      <mesh position={[0.12, 0.28, 0]} rotation={[0, 0, -0.55]}>
        <boxGeometry args={[0.09, 0.09, 0.56]} />
        <meshStandardMaterial color="#c7cbc5" metalness={0.6} roughness={0.28} />
      </mesh>
      <mesh position={[0.36, 0.47, 0]}>
        <boxGeometry args={[0.25, 0.1, 0.16]} />
        <meshStandardMaterial color="#202827" metalness={0.45} />
      </mesh>
    </group>
  );
}

function explodedPosition(position: Vec3, index: number): Vec3 {
  const angle = index * 2.39996;
  const distance = 0.45 + index * 0.18;
  return {
    x: position.x + Math.cos(angle) * distance,
    y: position.y + Math.sin(angle) * distance,
    z: position.z + (index % 5) * 0.16
  };
}
