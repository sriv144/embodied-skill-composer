import { ExternalLink } from "lucide-react";
import { api } from "../api";
import type { ArtifactReference } from "../types";

export function EvidenceReference({
  path,
  label = "Open artifact"
}: {
  path: string | null | undefined;
  label?: string;
}) {
  if (!path) return <span className="muted-line">No artifact registered</span>;
  const href = api.artifactHref(path);
  if (!href) {
    return (
      <span className="muted-line">
        Artifact link unavailable in this runtime
      </span>
    );
  }
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {label} <ExternalLink aria-hidden="true" size={12} />
    </a>
  );
}

export function EvidenceReferences({
  references
}: {
  references: ArtifactReference[];
}) {
  if (references.length === 0) {
    return (
      <span className="muted-line">No downloadable evidence registered</span>
    );
  }
  return (
    <ul className="evidence-links" aria-label="Downloadable evidence artifacts">
      {references.map((reference) => {
        const href = api.artifactHref(reference.href);
        if (!href) return null;
        return (
          <li key={`${reference.path}-${reference.href}`}>
            <a href={href} target="_blank" rel="noreferrer">
              {reference.label} <ExternalLink aria-hidden="true" size={12} />
            </a>
          </li>
        );
      })}
    </ul>
  );
}
