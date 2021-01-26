if(!require(ordinal)) {
    install.packages("ordinal", dependencies=TRUE)
}
library(ordinal)
if(!require(car)) {
    install.packages("car", dependencies=TRUE)
}
library(car)
if(!require(RVAideMemoire)) {
    install.packages("RVAideMemoire", dependencies=TRUE)
}
library(RVAideMemoire)
if(!require(emmeans)) {
    install.packages("emmeans", dependencies=TRUE)
}
library(emmeans)
library(stringr)

args <- commandArgs(trailingOnly = TRUE)

filename <- args[1]
score_name <- args[2]
is_crossed <- FALSE

mode_info <- unlist(str_split(args[3], ":"))

effects_mode <- mode_info[1]

if (effects_mode == "crossed") {
	is_crossed <- TRUE
}

adjust <- "tukey"
if (length(mode_info) > 1) {
	adjust <- mode_info[2]
}

outfile <- NULL
if (length(args) == 4) {
	outfile <- args[4]
}


data <- read.csv(file=filename)

data$annotator <- factor(data$annotator)
data$document <- factor(data$document)
data$system <- factor(data$system)

scores <- data[score_name]

if (score_name == "rank") {
	scores <- -scores
}

data$score <- factor(unlist(scores))

if (is_crossed) {
	model <- clmm(score ~ system + (system|annotator) + (system|document), data=data)
} else {
	model <- clmm(score ~ system + (system|annotator), data=data)
}

if (!is.null(outfile)) {
	random_effects <- VarCorr(model)
	coefficients <- model$beta
	coeff_names <- unlist(lapply(names(coefficients), function(n) { substring(n, 7) }))
	system_names <- c("__REFERENCE__", coeff_names)
	thresholds <- unname(model$alpha)
	coefficients <- c(0.0, unname(coefficients))

	library(rjson)
	model_json = toJSON(list(
		random_effects=random_effects,
		coefficients=coefficients,
		system_names=system_names,
		thresholds=thresholds
	))

	write(model_json, outfile)
}


marginal <- emmeans(model, "system")
marginal <- pairs(marginal, infer=c(TRUE, TRUE), adjust=adjust)
    marginal <- as.data.frame(marginal)

#sig_diffs <- marginal[marginal["p.value"] < 0.05,]

for (idx in 1:nrow(marginal)) {
    sign <- "o"
    if (marginal[idx, "p.value"] < 0.05) {
        sign <- "+"
        if (marginal[idx, "estimate"] < 0.0) {
            sign <- "-"
        }
    }
    cat(unlist(marginal[idx, "contrast"])[1])
    cat("\t")
    cat(sign)
    cat("\t")
    cat(marginal[idx, "p.value"])
    cat("\n")
}
