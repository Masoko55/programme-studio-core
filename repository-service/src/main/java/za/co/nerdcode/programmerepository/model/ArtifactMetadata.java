package za.co.nerdcode.programmerepository.model;

public record ArtifactMetadata(
    String artifactId,
    String referenceNumber,
    String artifactKey,
    String mimeType,
    String sha256,
    long size,
    String downloadUrl
) {
}