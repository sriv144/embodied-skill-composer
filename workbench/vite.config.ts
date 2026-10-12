import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import {
  developmentSecurityHeaders,
  previewSecurityHeaders
} from "./src/securityHeaders";

const loopbackProxy = {
  "/api": {
    target: "http://127.0.0.1:8008",
    ws: true
  },
  "/artifacts": "http://127.0.0.1:8008"
};

export default defineConfig({
  base: "./",
  plugins: [react()],
  resolve: {
    alias: [
      {
        find: /^three$/,
        replacement: fileURLToPath(
          new URL("./node_modules/three/src/Three.js", import.meta.url)
        )
      }
    ]
  },
  server: {
    headers: developmentSecurityHeaders,
    port: 5173,
    proxy: loopbackProxy
  },
  preview: {
    headers: previewSecurityHeaders,
    proxy: loopbackProxy
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id, meta) {
          if (!id.includes("node_modules")) return undefined;
          const moduleId = id.replaceAll("\\", "/");
          if (moduleId.includes("@react-three/drei")) return "three-drei";
          if (moduleId.includes("@react-three/fiber")) return "three-fiber";
          if (moduleId.includes("@xyflow/react")) return "xyflow";
          if (moduleId.includes("framer-motion")) return "motion";
          if (moduleId.includes("/three/examples/")) return "three-extras";
          if (moduleId.includes("/three/src/")) {
            const depthCache = new Map<string, number>();
            const dependencyDepth = (
              candidate: string,
              visiting = new Set<string>()
            ): number => {
              const cached = depthCache.get(candidate);
              if (cached !== undefined) return cached;
              if (visiting.has(candidate)) return 0;
              const nextVisiting = new Set(visiting).add(candidate);
              const info = meta.getModuleInfo(candidate);
              const dependencies = info?.importedIds.filter((dependency) =>
                dependency.replaceAll("\\", "/").includes("/three/src/")
              ) ?? [];
              const depth =
                dependencies.length === 0
                  ? 0
                  : 1 +
                    Math.max(
                      ...dependencies.map((dependency) =>
                        dependencyDepth(dependency, nextVisiting)
                      )
                    );
              depthCache.set(candidate, depth);
              return depth;
            };
            return `three-layer-${Math.floor(dependencyDepth(id) / 3)}`;
          }
          if (/\/react(?:-dom)?\//.test(moduleId)) return "react";
          return undefined;
        }
      }
    }
  }
});
