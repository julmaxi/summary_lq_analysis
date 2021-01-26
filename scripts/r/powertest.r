library(simr)
library(lme4)
library(lmerTest)

annotations <- read.csv("~/Documents/work/summary_repository/20200506-182938.csv")

filtered_annotations <- annotations[annotations$system == "BART" | annotations$system == "__REFERENCE__",]

filtered_annotations$system <- factor(filtered_annotations$system)
filtered_annotations$annotator <- factor(filtered_annotations$annotator)
filtered_annotations$instance <- factor(filtered_annotations$instance)


#filtered_annotations$system[filtered_annotations$system == 0] = -1
#filtered_annotations$system[filtered_annotations$system == 1] = +1

model <- lmer(coherence_score ~ system + (system | annotator) + (system | instance) + (1 | annotator:instance), data=filtered_annotations)
fixef(model)["systemBART"] <- 0.5

#powerSim(model, nsim=100)

create_nested_instances <- function (n_annotators, n_annotations) {
    result <- vector()
    for (i in 1:n_annotators) {
        new_vals <- rep(
            (1 + (i - 1) * n_annotations):(n_annotations * i), times=2
        )

        result <- c(result, new_vals)
    }
    return(result)
}

n_instances = 10

#instance <- factor(1:10)
instance_nested <- create_nested_instances(5, n_instances)
annotator_nested <- factor(rep(c(1:5), each=n_instances * 2))

instance <- factor(
    rep(1:n_instances, rep=2)
)
annotator <- factor(
    rep(1:5,each=n_instances * 2)
)

system <- factor(rep(c(rep("A", times=n_instances), rep("B", times=n_instances))))

X <- data.frame(instance, annotator, system)
X_easy <- data.frame(system, instance=instance_nested, annotator=annotator_nested)

print(X)

b <- c(3.55, 0.5) # fixed intercept and slope
V1 <- 0.4 # random intercept variance
V2 <- matrix(c(0.2,-0.1,-0.1,0.06), 2) # random intercept and slope variance-covariance matrix
s <- 0.6 # residual standard deviation

simple_model <- makeLmer(y ~ system + (system|annotator) + (1|instance), fixef=b, VarCorr=list(V1, V2), sigma=s, data=X_easy)
complex_model <- makeLmer(y ~ system + (system | annotator) + (system | instance) + (1 | annotator:instance), fixef=b, VarCorr=VarCorr(model), sigma=s, data=X)

summary(model)

complex_result <- powerSim(complex_model, nsim=1000)
simple_result <- powerSim(simple_model, nsim=1000)


print(complex_result)
print(simple_result)


#summary(model)