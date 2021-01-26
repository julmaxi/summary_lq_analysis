library(simr)
library(lme4)
library(lmerTest)
library(iterators)
library(ordinal)


annotations <- read.csv("large_results_crossed.csv")

annotations$system_f <- factor(annotations$system)
annotations$annotator_f <- factor(annotations$annotator)
annotations$instance_f <- factor(annotations$document)
annotations$coherence_f <- factor(annotations$coherence_score)


systems <- unique(annotations$system)

results <- sapply(
    combn(systems, 2, simplify=FALSE),
    function(system_pair) {
    filtered_annotations <- annotations[annotations$system == system_pair[1] | annotations$system == system_pair[2],]
    print(system_pair)
    #model <- lmer(coherence_score ~ system_f + (system_f | annotator_f) + (system_f | instance_f), data=filtered_annotations)
    model <- clmm(coherence_f ~ system_f + (1|annotator_f) + (1|instance_f), data=filtered_annotations)
    anova(model)
    #c(system_pair, anova(model)$Pr)
    }
)

print(results)

#summary(model)
#fixef(model)["system1"] <- 0.5
#obs <- matrix(1:20, nrow=5, ncol=5)
#
#print(powerSim(model, nsim=100))

quit()


anno_idx <- 1

for (n_annotators in c(3, 4, 5, 10, 20)) {
    inst_idx <- 1
    for (n_instances in c(20, 50, 100, 200, 500)) {
        new_model <- extend(model, along="instance_f", n=n_instances)
        new_model <- extend(new_model, along="annotator_f", n=n_annotators)
        sim_results <- powerSim(new_model, nsim=100, progress=FALSE)
        obs[anno_idx, inst_idx] <- summary(sim_results)[1,"mean"]
        inst_idx <- inst_idx + 1
    }
    anno_idx <- anno_idx + 1
}

print(obs)
