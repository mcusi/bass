library(tidyverse)
library(patchwork)
library(png)
library(ggpattern)
library(progressr)
library(boot)
library(furrr)
library(knitr)
library(kableExtra)

model_names <- c(
  "baseline"="Baseline",
  "fuss-pretrain"= "TCDN++\nFUSS",
  "Libri2Mix-sepclean"=  "ConvTasNet\nLibri2Mix-clean",
  "Libri2Mix-sepnoisy"= "ConvTasNet\nLibri2Mix-noisy",
  "mixit-pretrain"= "MixIT\nYFCC100M",
  "gen-model"= "(Ours) Gen. model\nSequential only",
  "enumerative-full"= "(Ours) Gen. model\nEnum. + Seq.",
  "enumerative-map"= "Fixed",
  "enumerative-stationary" = "Stationary",
  "enumerative-spectralswap" = "Spectral swap",
  "enumerative-uniform" = "Uniform",
  "demucs" = "HT Demucs\nMUSDB + 800",
  "sepformer" = "SepFormer\nWHAMR!"
)

model_colors <- c(
  "baseline"="black",
  "network"="darkgray",
  "amortized"="darkgray",
  "sequential"="#CD3278",
  "enumerative-map"="#7E72B4",
  "enumerative-uniform"="#B6A9D2",
  "enumerative-spectralswitch"="#3557A6",
  "enumerative-full"="#4BA5DB")

# Load results for all models & demos ####
df <- read_csv("xy_pairs.csv") %>%
  mutate(permutation=replace_na(permutation, 0)) %>%
  filter(demo != "spectral") %>% # spectral1 and spectral2 separately
  filter(!str_detect(model, "umxhq")) %>%
  mutate(demo = ifelse(demo %in% c("masking", "continuity"), "mc", demo)) %>%
  mutate(demo_idx = as.integer(as.factor(demo))) %>%
  group_by(demo, model, permutation) %>%
  mutate(xy_idx = row_number()) %>%
  ungroup()

df_model_names <- df %>%
  group_by(model) %>%
  summarize() %>%
  mutate(model_name = ifelse(str_detect(model, "fuss-amortize"), "TCDN++\nGenerated Samples", coalesce(model_names[model], model))) %>%
  mutate(model_type = case_when(
    str_detect(model, "enumerative") ~ model,
    str_detect(model, "baseline") ~ "baseline",
    model=="gen-model" ~ "sequential",
    str_detect(model, "amortize") ~ "amortized",
    T ~ "network"
  ) %>% fct_relevel("network", "amortized", "sequential"))





# Run bootstrap ####
R <- 1000 # Number of bootstrap samples
# List of all demos and experimental conditions for making judgements
df_base <- df %>%
  filter(model=="enumerative-full", permutation==0) %>%
  select(demo, demo_idx, xy_idx) 

calc_errs <- function(d, return_demo_errs=F) {
  # This function takes as input a dataframe d with columns:
  # - demo, demo_idx
  # - model
  # - permutation: for correlation type 1, assignment of model outputs to standards
  # - xy_idx: the experimental condition on which a judgement is made
  # - x: The model's judgement
  # - y: The human's judgement
  
  # The output is a dataframe with one row per model, and columns:
  # - aba, abruptness_abrupt, abruptness_slow, etc... (the r for each demo)
  # - err: the error calculated from this r (compared vs. the baseline model)
  # - diff_enumerative, diff_sequential: the error relative to the Sequential/Enumerative Full models
  
  # Calculate correlations
  corrs <- d %>%
    group_by(model, permutation, demo) %>%
    group_modify(function(d, k) {
      use_spearman <- k$demo %in% c("mc", "cmr", "mistuned_harmonic", "onset_asynchrony")
      r <- if((n_distinct(d$x)==1) | (n_distinct(d$y)==1)) 0 else cor(d$x, d$y, method=if(use_spearman) "spearman" else "pearson")
      tibble(r=r)
    }) %>%
    ungroup() %>%
    group_by(model, demo) %>%
      slice_max(r, with_ties = F) %>% # When there's multiple permutations, choose the best
    ungroup() %>%
    select(-permutation)
  
  if(return_demo_errs) {
    errs <- left_join(corrs, corrs %>% filter(model=="baseline"), by="demo", suffix=c("", ".baseline")) %>%
             mutate(err=ifelse(r<=r.baseline, 1, (1-r) / (1-r.baseline))) 
    return(errs)
  }
  
  # Calculate error = (1-r)/(1-r_baseline), averaged over all demos
  errs <- left_join(corrs, corrs %>% filter(model=="baseline"), by="demo", suffix=c("", ".baseline")) %>%
    mutate(err=ifelse(r<=r.baseline, 1, (1-r) / (1-r.baseline))) %>%
    group_by(model) %>%
      summarize(err = mean(err), .groups="drop") %>%
    ungroup()
  
  # For each model we return:
  # - the correlation on every demo
  # - the error
  # - the error relative to the Sequential/Enumerative Full models
  ref_enumerative <- errs %>% filter(model=="enumerative-full") %>% pull(err) %>% min
  ref_sequential <- errs %>% filter(model=="gen-model") %>% pull(err) %>% min
  errs %>%
    left_join(corrs %>% pivot_wider(names_from=demo, values_from=r), by=c("model"), .) %>%
    mutate(diff_enumerative = err - ref_enumerative,
           diff_sequential = err - ref_sequential)
}

# Calculate errors
errs <- calc_errs(df)
errs_long <- errs %>% pivot_longer(-c(model)) %>% select(model, name, value)

# Run the bootstrap!
# For each bootstrap resample, we save all the correlations and overall errors
handlers(handler_progress(format   = ":spin :current/:total [:bar] :percent in :elapsed ETA: :eta"))
b <- with_progress({p = progressor(steps=R); boot(data=df_base, R=R, strata=df_base$demo_idx, statistic = function(original, i) {
  d <- df_base[i,] %>% select(demo, xy_idx) %>% left_join(df, by=c("demo", "xy_idx"))
  errs <- calc_errs(d)
  errs_long <- errs %>% pivot_longer(-c(model))
  
  p()
  errs_long %>% pull(value)
})})
boot_samples <- as_tibble(b$t) %>%
  set_names(paste0(errs_long$model, "___", errs_long$name)) %>%
  mutate(boot_idx = row_number()) %>%
  pivot_longer(-boot_idx, names_to = c("model", "name"), names_sep = "___")
boot_summary <- boot_samples %>% # Bootstrap statistics
  group_by(model, name) %>%
  summarize(average=mean(value), std.error=sd(value), .groups="drop") %>%
  left_join(errs_long %>% rename(estimate=value))

models = boot_summary %>%
  filter(name == "err") %>%
  left_join(df_model_names) %>%
  group_by(model_name, name) %>%
    slice_min(estimate, with_ties=F) %>% # TCDN++\nGenerated Samples was run with many parameter settings. Choose the best.
  ungroup() %>%
  select(model)

df_results <- models %>%
  left_join(boot_summary) %>%
  left_join(df_model_names)


# Calculate 99% Basic bootstrap interval for every model's error relative to enumerative-full
# (Difference is not significant for enumerative-uniform or gen-model)
conf <- 0.99
cis <- errs_long %>%
  mutate(i = row_number(), keep=!is.na(b$t0)) %>%
  filter(keep) %>%
  pmap_dfr(function(i, keep, model, name, value) {
    ci <- boot.ci(b, index=i, type = c("basic"), conf = conf)
    tibble(model, name, value, conf.low = ci$basic[[4]], conf.high = ci$basic[[5]])  
  })

cis %>%
  filter(name=="diff_enumerative") %>%
  filter(! model %in% c("baseline", "enumerative-stationary", "enumerative-spectralswap")) %>%
  mutate(model = fct_reorder(model, value)) %>%
  ggplot(aes(y=model, x=value)) +
  geom_vline(xintercept=0, linetype="dashed") +
  geom_linerange(aes(xmin=conf.low, xmax=conf.high), alpha=0.5) +
  geom_point() +
  theme_bw() + theme(
    panel.grid.major.y = element_blank()
  ) +
  labs(x="Error relative to enumerative-full (99% CI)")




# Plots ####
plot_data <- df_results %>%
  filter(name=="err") %>%
  mutate(err=estimate) %>%
  select(model, err, std.error) %>%
  left_join(df_model_names) %>%
  mutate(model_name = model_name %>% fct_reorder(-err) %>% fct_reorder(model=="enumerative-full")) %>%
  filter(!str_detect(model_name, "Open-Unmix"))

g_network <- plot_data  %>%
  mutate(pattern=ifelse(model_type %in% c("network"), 'stripe', 'none')) %>%
  filter(model_type %in% c("network", "amortized", "sequential", "enumerative-full")) %>%
  mutate(model_name = fct_reorder(model_name, as.integer(model_type))) %>%
  ggplot(aes(x=model_name, y=err, fill=model_type, pattern=pattern)) +
  scale_fill_manual(values = model_colors) +
  scale_pattern_manual(values=c('stripe'='stripe', 'none'='none')) +
  geom_bar_pattern(stat='identity', alpha=0.9, color="black", size=0.4, pattern_alpha=0.3, pattern_density=0.5) +
  geom_errorbar(aes(ymin=err-std.error, ymax=err+std.error), width=0.1) +
  theme_bw() + theme(
    panel.grid.major.x = element_blank(),
    strip.background = element_blank(),
    legend.position = "none",
    axis.text.x = element_text(angle=45, hjust=1, vjust=1)
  ) +
  coord_cartesian(ylim=c(0,1)) +
  labs(x="", y="Dissimilarity") +
  ggtitle("A. Neural networks")
g_network

g_map_uniform <- plot_data  %>%
  filter(model %in% c("enumerative-uniform", "enumerative-map", "enumerative-full")) %>%
  ggplot(aes(x=model_name, y=err, fill=model_type)) +
  scale_fill_manual(values = model_colors) +
  geom_bar(stat='identity', alpha=0.9, color="black", size=0.4) +
  geom_errorbar(aes(ymin=err-std.error, ymax=err+std.error), width=0.1) +
  theme_bw() + theme(
    panel.grid.major.x = element_blank(),
    strip.background = element_blank(),
    legend.position = "none",
    axis.text.x = element_text(angle=45, hjust=1, vjust=1),
    axis.text.y = element_blank(),
    axis.title.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  coord_cartesian(ylim=c(0,1)) +
  labs(x="", y="Dissimilarity") +
  ggtitle("B. Hyperpriors")
g_map_uniform

g <- g_network + g_map_uniform +
  plot_layout(widths=c(3.3,1))

g
ggsave("graphs.pdf", g, width=8, height=5)