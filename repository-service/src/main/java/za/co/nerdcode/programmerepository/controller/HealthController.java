package za.co.nerdcode.programmerepository.controller;

import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HealthController {

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of(
            "status",
            "healthy",
            "service",
            "repository-service"
        );
    }

    @GetMapping("/ready")
    public Map<String, String> ready() {
        return Map.of(
            "status",
            "ready",
            "service",
            "repository-service"
        );
    }
}