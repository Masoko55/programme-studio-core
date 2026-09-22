package za.co.nerdcode.programmerepository.service;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.UUID;

import javax.jcr.Binary;
import javax.jcr.Credentials;
import javax.jcr.Node;
import javax.jcr.Repository;
import javax.jcr.Session;
import javax.jcr.SimpleCredentials;

import org.springframework.stereotype.Service;

import za.co.nerdcode.programmerepository.model.ArtifactMetadata;

@Service
public class RepositoryStorageService {

    private final Repository repository;

    private final Credentials credentials =
        new SimpleCredentials(
            "admin",
            "admin".toCharArray()
        );

    public RepositoryStorageService(
        Repository repository
    ) {
        this.repository = repository;
    }

    private Session login() throws Exception {
        return repository.login(
            credentials
        );
    }

    private String calculateSha256(
        byte[] content
    ) throws Exception {
        MessageDigest digest =
            MessageDigest.getInstance(
                "SHA-256"
            );

        return HexFormat
            .of()
            .formatHex(
                digest.digest(content)
            );
    }

    private String sanitizeNodeName(
        String value
    ) {
        return value
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_");
    }

    private Node getOrCreateChild(
        Node parent,
        String name
    ) throws Exception {
        if (parent.hasNode(name)) {
            return parent.getNode(name);
        }

        return parent.addNode(
            name,
            "nt:unstructured"
        );
    }

    public ArtifactMetadata saveArtifact(
        String referenceNumber,
        String artifactKey,
        String mimeType,
        String expectedSha256,
        byte[] content
    ) throws Exception {

        String actualSha256 =
            calculateSha256(content);

        if (
            expectedSha256 != null
            && !expectedSha256.isBlank()
            && !expectedSha256.equalsIgnoreCase(
                actualSha256
            )
        ) {
            throw new IllegalArgumentException(
                "SHA-256 mismatch."
            );
        }

        Session session = login();

        try {
            Node root =
                session.getRootNode();

            Node references =
                getOrCreateChild(
                    root,
                    "references"
                );

            Node reference =
                getOrCreateChild(
                    references,
                    sanitizeNodeName(
                        referenceNumber
                    )
                );

            Node artifacts =
                getOrCreateChild(
                    reference,
                    "artifacts"
                );

            String artifactId =
                UUID.randomUUID()
                    .toString();

            Node artifact =
                artifacts.addNode(
                    artifactId,
                    "nt:unstructured"
                );

            artifact.setProperty(
                "artifactId",
                artifactId
            );

            artifact.setProperty(
                "referenceNumber",
                referenceNumber
            );

            artifact.setProperty(
                "artifactKey",
                artifactKey
            );

            artifact.setProperty(
                "mimeType",
                mimeType
            );

            artifact.setProperty(
                "sha256",
                actualSha256
            );

            artifact.setProperty(
                "size",
                content.length
            );

            Binary binary =
                session
                    .getValueFactory()
                    .createBinary(
                        new ByteArrayInputStream(
                            content
                        )
                    );

            artifact.setProperty(
                "data",
                binary
            );

            session.save();

            return new ArtifactMetadata(
                artifactId,
                referenceNumber,
                artifactKey,
                mimeType,
                actualSha256,
                content.length,
                "/v1/artifacts/"
                    + artifactId
            );

        } finally {
            session.logout();
        }
    }

    public void saveManifest(
        String referenceNumber,
        String manifestJson
    ) throws Exception {

        Session session = login();

        try {
            Node root =
                session.getRootNode();

            Node references =
                getOrCreateChild(
                    root,
                    "references"
                );

            Node reference =
                getOrCreateChild(
                    references,
                    sanitizeNodeName(
                        referenceNumber
                    )
                );

            reference.setProperty(
                "manifest",
                manifestJson
            );

            session.save();

        } finally {
            session.logout();
        }
    }

    public String loadManifest(
        String referenceNumber
    ) throws Exception {

        Session session = login();

        try {
            Node root =
                session.getRootNode();

            String referencePath =
                "references/"
                + sanitizeNodeName(
                    referenceNumber
                );

            if (
                !root.hasNode(
                    referencePath
                )
            ) {
                throw new IllegalArgumentException(
                    "Reference not found."
                );
            }

            Node reference =
                root.getNode(
                    referencePath
                );

            if (
                !reference.hasProperty(
                    "manifest"
                )
            ) {
                throw new IllegalArgumentException(
                    "Manifest not found."
                );
            }

            return reference
                .getProperty(
                    "manifest"
                )
                .getString();

        } finally {
            session.logout();
        }
    }

    public List<ArtifactMetadata> listArtifacts(
        String referenceNumber
    ) throws Exception {

        List<ArtifactMetadata> results =
            new ArrayList<>();

        Session session = login();

        try {
            Node root =
                session.getRootNode();

            String path =
                "references/"
                + sanitizeNodeName(
                    referenceNumber
                )
                + "/artifacts";

            if (!root.hasNode(path)) {
                return results;
            }

            Node artifacts =
                root.getNode(path);

            var nodes =
                artifacts.getNodes();

            while (nodes.hasNext()) {
                Node artifact =
                    nodes.nextNode();

                results.add(
                    new ArtifactMetadata(
                        artifact.getProperty(
                            "artifactId"
                        ).getString(),

                        artifact.getProperty(
                            "referenceNumber"
                        ).getString(),

                        artifact.getProperty(
                            "artifactKey"
                        ).getString(),

                        artifact.getProperty(
                            "mimeType"
                        ).getString(),

                        artifact.getProperty(
                            "sha256"
                        ).getString(),

                        artifact.getProperty(
                            "size"
                        ).getLong(),

                        "/v1/artifacts/"
                            + artifact.getProperty(
                                "artifactId"
                            ).getString()
                    )
                );
            }

            return results;

        } finally {
            session.logout();
        }
    }

    public byte[] loadArtifact(
        String artifactId
    ) throws Exception {

        Session session = login();

        try {
            Node root =
                session.getRootNode();

            if (!root.hasNode(
                "references"
            )) {
                throw new IllegalArgumentException(
                    "Artifact not found."
                );
            }

            Node references =
                root.getNode(
                    "references"
                );

            var referenceNodes =
                references.getNodes();

            while (
                referenceNodes.hasNext()
            ) {
                Node reference =
                    referenceNodes.nextNode();

                if (
                    !reference.hasNode(
                        "artifacts"
                    )
                ) {
                    continue;
                }

                Node artifacts =
                    reference.getNode(
                        "artifacts"
                    );

                if (
                    artifacts.hasNode(
                        artifactId
                    )
                ) {
                    Node artifact =
                        artifacts.getNode(
                            artifactId
                        );

                    Binary binary =
                        artifact
                            .getProperty(
                                "data"
                            )
                            .getBinary();

                    try (
                        InputStream stream =
                            binary.getStream()
                    ) {
                        return stream
                            .readAllBytes();
                    }
                }
            }

            throw new IllegalArgumentException(
                "Artifact not found."
            );

        } finally {
            session.logout();
        }
    }

    public String getArtifactMimeType(
        String artifactId
    ) throws Exception {

        Session session = login();

        try {
            Node root =
                session.getRootNode();

            if (!root.hasNode(
                "references"
            )) {
                throw new IllegalArgumentException(
                    "Artifact not found."
                );
            }

            Node references =
                root.getNode(
                    "references"
                );

            var referenceNodes =
                references.getNodes();

            while (
                referenceNodes.hasNext()
            ) {
                Node reference =
                    referenceNodes.nextNode();

                if (
                    !reference.hasNode(
                        "artifacts"
                    )
                ) {
                    continue;
                }

                Node artifacts =
                    reference.getNode(
                        "artifacts"
                    );

                if (
                    artifacts.hasNode(
                        artifactId
                    )
                ) {
                    return artifacts
                        .getNode(
                            artifactId
                        )
                        .getProperty(
                            "mimeType"
                        )
                        .getString();
                }
            }

            throw new IllegalArgumentException(
                "Artifact not found."
            );

        } finally {
            session.logout();
        }
    }
}