package za.co.nerdcode.programmerepository.controller;

import java.util.List;
import java.util.Map;

import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import za.co.nerdcode.programmerepository.model.ArtifactMetadata;
import za.co.nerdcode.programmerepository.service.RepositoryStorageService;

@RestController
@RequestMapping("/v1")
public class RepositoryController {

    private final RepositoryStorageService storageService;

    public RepositoryController(
        RepositoryStorageService storageService
    ) {
        this.storageService =
            storageService;
    }

    @PutMapping(
        value =
            "/references/{referenceNumber}/artifacts/{artifactKey}",
        consumes = {
            MediaType.APPLICATION_OCTET_STREAM_VALUE,
            MediaType.IMAGE_PNG_VALUE
        }
    )
    public ResponseEntity<?> uploadArtifact(
        @PathVariable
        String referenceNumber,

        @PathVariable
        String artifactKey,

        @RequestHeader(
            value = "X-SHA256",
            required = false
        )
        String sha256,

        @RequestHeader(
            value = HttpHeaders.CONTENT_TYPE,
            required = false
        )
        String contentType,

        @RequestBody
        byte[] content
    ) {
        try {
            String mimeType =
                contentType == null
                    ? MediaType
                        .APPLICATION_OCTET_STREAM_VALUE
                    : contentType;

            ArtifactMetadata result =
                storageService.saveArtifact(
                    referenceNumber,
                    artifactKey,
                    mimeType,
                    sha256,
                    content
                );

            return ResponseEntity.ok(
                result
            );

        } catch (
            IllegalArgumentException error
        ) {
            return ResponseEntity
                .badRequest()
                .body(
                    Map.of(
                        "error",
                        error.getMessage()
                    )
                );

        } catch (Exception error) {
            return ResponseEntity
                .internalServerError()
                .body(
                    Map.of(
                        "error",
                        error.getMessage()
                    )
                );
        }
    }

    @PutMapping(
        value =
            "/references/{referenceNumber}/manifest",
        consumes =
            MediaType.APPLICATION_JSON_VALUE
    )
    public ResponseEntity<?> uploadManifest(
        @PathVariable
        String referenceNumber,

        @RequestBody
        String manifestJson
    ) {
        try {
            storageService.saveManifest(
                referenceNumber,
                manifestJson
            );

            return ResponseEntity.ok(
                Map.of(
                    "status",
                    "stored",
                    "reference_number",
                    referenceNumber
                )
            );

        } catch (Exception error) {
            return ResponseEntity
                .internalServerError()
                .body(
                    Map.of(
                        "error",
                        error.getMessage()
                    )
                );
        }
    }

    @GetMapping(
        value =
            "/references/{referenceNumber}/manifest",
        produces =
            MediaType.APPLICATION_JSON_VALUE
    )
    public ResponseEntity<?> getManifest(
        @PathVariable
        String referenceNumber
    ) {
        try {
            String manifest =
                storageService.loadManifest(
                    referenceNumber
                );

            return ResponseEntity
                .ok()
                .contentType(
                    MediaType.APPLICATION_JSON
                )
                .body(
                    manifest
                );

        } catch (
            IllegalArgumentException error
        ) {
            return ResponseEntity
                .notFound()
                .build();

        } catch (Exception error) {
            return ResponseEntity
                .internalServerError()
                .body(
                    Map.of(
                        "error",
                        error.getMessage()
                    )
                );
        }
    }

    @GetMapping(
        "/references/{referenceNumber}/artifacts"
    )
    public ResponseEntity<?> listArtifacts(
        @PathVariable
        String referenceNumber
    ) {
        try {
            List<ArtifactMetadata> artifacts =
                storageService.listArtifacts(
                    referenceNumber
                );

            return ResponseEntity.ok(
                artifacts
            );

        } catch (Exception error) {
            return ResponseEntity
                .internalServerError()
                .body(
                    Map.of(
                        "error",
                        error.getMessage()
                    )
                );
        }
    }

    @GetMapping(
        "/artifacts/{artifactId}"
    )
    public ResponseEntity<?> downloadArtifact(
        @PathVariable
        String artifactId
    ) {
        try {
            byte[] content =
                storageService.loadArtifact(
                    artifactId
                );

            String mimeType =
                storageService
                    .getArtifactMimeType(
                        artifactId
                    );

            return ResponseEntity
                .ok()
                .contentType(
                    MediaType.parseMediaType(
                        mimeType
                    )
                )
                .body(content);

        } catch (
            IllegalArgumentException error
        ) {
            return ResponseEntity
                .notFound()
                .build();

        } catch (Exception error) {
            return ResponseEntity
                .internalServerError()
                .body(
                    Map.of(
                        "error",
                        error.getMessage()
                    )
                );
        }
    }
}
