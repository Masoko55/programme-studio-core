package za.co.nerdcode.programmerepository.config;

import java.io.File;

import javax.jcr.Repository;

import org.apache.jackrabbit.oak.Oak;
import org.apache.jackrabbit.oak.jcr.Jcr;
import org.apache.jackrabbit.oak.segment.SegmentNodeStore;
import org.apache.jackrabbit.oak.segment.SegmentNodeStoreBuilders;
import org.apache.jackrabbit.oak.segment.file.FileStore;
import org.apache.jackrabbit.oak.segment.file.FileStoreBuilder;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;


@Configuration
public class OakConfiguration {

    @Value("${repository.path}")
    private String repositoryPath;


    @Bean(destroyMethod = "close")
    public FileStore fileStore()
        throws Exception {

        File directory = new File(
            repositoryPath
        );

        if (!directory.exists()) {
            boolean created =
                directory.mkdirs();

            if (!created) {
                throw new IllegalStateException(
                    "Could not create repository directory: "
                    + directory.getAbsolutePath()
                );
            }
        }

        return FileStoreBuilder
            .fileStoreBuilder(
                directory
            )
            .build();
    }


    @Bean
    public SegmentNodeStore nodeStore(
        FileStore fileStore
    ) {

        return SegmentNodeStoreBuilders
            .builder(
                fileStore
            )
            .build();
    }


    @Bean
    public Repository repository(
        SegmentNodeStore nodeStore
    ) {

        return new Jcr(
            new Oak(
                nodeStore
            )
        ).createRepository();
    }
}