#!/usr/bin/env Rscript
# =============================================================================
# run_perez_priego.R
# =============================================================================
# Runs the Pérez-Priego ET partitioning method on preprocessed OzFlux L6 data.
#
# Usage:
#   Rscript run_perez_priego.R [site_name]
# =============================================================================

# Get script directory robustly
args_all <- commandArgs(trailingOnly = FALSE)
script_arg <- args_all[grep("--file=", args_all)]
if (length(script_arg) > 0) {
  SCRIPT_DIR <- dirname(sub("--file=", "", script_arg))
} else {
  SCRIPT_DIR <- "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition"
}
OUTPUT_DIR <- file.path(SCRIPT_DIR, "output")
if (!dir.exists(OUTPUT_DIR)) {
  OUTPUT_DIR <- "/home/sanjays/et97_scratch2/oldscratch/Ozflux_data_full/TEA_partition/output"
}

library(ncdf4)
library(ETpartitioning)
library(FME)

# =============================================================================
# Functions
# =============================================================================

run_pp_site <- function(preproc_file, site_name, output_dir) {
  cat(sprintf("\n[Pérez-Priego] Processing: %s\n", site_name))
  
  # Open preprocessed NetCDF
  nc <- nc_open(preproc_file)
  
  # Extract variables
  time_vals <- ncvar_get(nc, "time")
  time_posix <- as.POSIXct(time_vals * 86400, origin="1858-11-17", tz="UTC") # Approximate for now, or just use as sequence
  # Wait, ncdf4 time is usually days since 1990 or similar, but the Python xarray writes it as ns since 1970
  # We will just generate an artificial time sequence since we only need date grouping.
  
  ET    <- ncvar_get(nc, "ET")       # mm/timestep
  LE    <- ncvar_get(nc, "LE")       # W m-2
  H     <- ncvar_get(nc, "H")        # W m-2
  NEE   <- ncvar_get(nc, "NEE")      # umol m-2 s-1
  TA    <- ncvar_get(nc, "TA")       # degC
  VPD   <- ncvar_get(nc, "VPD")      # hPa
  NETRAD <- ncvar_get(nc, "NETRAD")  # W m-2
  SW_IN <- ncvar_get(nc, "SW_IN")    # W m-2
  PA    <- ncvar_get(nc, "PA")       # kPa
  USTAR <- ncvar_get(nc, "USTAR")   # m s-1
  WS    <- ncvar_get(nc, "WS")      # m s-1
  NIGHT <- ncvar_get(nc, "NIGHT")   # 0/1
  
  # GPP
  GPP   <- ncvar_get(nc, "GPP_NT")   # umol m-2 s-1
  
  nc_close(nc)
  
  # Create synthetic date for looping (assuming 48 steps per day)
  n <- length(ET)
  steps_per_day <- 48
  loop_idx <- rep(1:ceiling(n/steps_per_day), each=steps_per_day)[1:n]
  
  # Build data frame
  df <- data.frame(
    ET = ET,
    LE = LE,
    H = H,
    GPP = GPP,
    GPP_unc = pmax(abs(GPP)*0.2, 0.1, na.rm=TRUE), # Mock uncertainty
    TA = TA,
    VPD = VPD,
    PA = PA,
    SW_IN = SW_IN,
    PPFD_IN = SW_IN * 2.1,
    USTAR = USTAR,
    WS = WS,
    CO2 = 400, # Mock CO2
    NIGHT = NIGHT,
    loop = loop_idx
  )
  
  # Quality filter - set NAs
  # We require positive ET and GPP for the daytime optimization
  bad_idx <- !is.finite(df$GPP) | !is.finite(df$H) | !is.finite(df$VPD) | !is.finite(df$TA) | !is.finite(df$PA) | !is.finite(df$SW_IN) | !is.finite(df$USTAR) | !is.finite(df$WS)
  df[bad_idx, c("GPP", "H", "VPD", "TA", "PA", "SW_IN", "USTAR", "WS")] <- NA
  
  # Altitude mock (0.1 km)
  Z <- 0.1
  
  tryCatch({
    # Calculate Chi_o and WUE_o using package functions
    Chi_o <- calculate_chi_o(data=df, ColPhotos="GPP", ColVPD="VPD", ColTair="TA", C=1.189, Z=Z)
    WUE_o <- calculate_WUE_o(data=df, ColPhotos="GPP", ColVPD="VPD", ColTair="TA", C=1.189, Z=Z)
    
    days_to_run <- unique(df$loop)
    estimation_out <- list()
    
    # Process only every 5th day or just a subset to save time if needed, but we'll try all
    # For performance, we'll loop but keep it efficient
    for (i in days_to_run) {
      tmp <- subset(df, loop %in% c(i-2, i-1, i, i+1, i+2))
      tmp_day <- subset(tmp, NIGHT == 0)
      
      # Remove NAs from the training subset
      tmp_day <- tmp_day[complete.cases(tmp_day[, c("GPP", "H", "VPD", "TA", "PA", "SW_IN", "USTAR", "WS")]), ]
      
      if (nrow(tmp_day) < 20) {
        # Not enough data to optimize
        tmp_central <- df[df$loop == i, ]
        tmp_central$T_PP <- NA
        estimation_out[[i]] <- tmp_central
        next
      }
      
      ans <- try(optimal_parameters(
        par_lower = c(0, 0, 10, 0),
        par_upper = c(400, 0.4, 30, 1),
        data = tmp_day,
        ColPhotos = "GPP",
        ColPhotos_unc = "GPP_unc",
        ColH = "H",
        ColVPD = "VPD",
        ColTair = "TA",
        ColPair = "PA",
        ColQ = "PPFD_IN",
        ColCa = "CO2",
        ColUstar = "USTAR",
        ColWS = "WS",
        ColSW_in = "SW_IN",
        Chi_o = Chi_o,
        WUE_o = WUE_o
      ), silent = TRUE)
      
      if (inherits(ans, "try-error")) {
        tmp_central <- df[df$loop == i, ]
        tmp_central$T_PP <- NA
        estimation_out[[i]] <- tmp_central
        next
      }
      
      par <- as.numeric(ans)
      
      # We must pass complete cases to transpiration_model to avoid errors, or handle NA
      # The package model might crash if NAs are present. 
      # So we predict only for valid rows
      valid_tmp <- complete.cases(tmp[, c("GPP", "H", "VPD", "TA", "PA", "SW_IN", "USTAR", "WS")])
      
      t_mod <- rep(NA, nrow(tmp))
      if (any(valid_tmp)) {
        t_mod_valid <- try(transpiration_model(
          par = par,
          data = tmp[valid_tmp, ],
          ColPhotos = "GPP",
          ColH = "H",
          ColVPD = "VPD",
          ColTair = "TA",
          ColPair = "PA",
          ColQ = "PPFD_IN",
          ColCa = "CO2",
          ColUstar = "USTAR",
          ColWS = "WS",
          ColSW_in = "SW_IN",
          Chi_o = Chi_o,
          WUE_o = WUE_o
        ), silent = TRUE)
        
        if (!inherits(t_mod_valid, "try-error")) {
          # Convert mmol m-2 s-1 to mm per half hour
          t_mod[valid_tmp] <- t_mod_valid * (18.01528 / 1e6) * 1800
        }
      }
      
      tmp$T_PP <- t_mod
      tmp_central <- tmp[tmp$loop == i, ]
      estimation_out[[i]] <- tmp_central
    }
    
    estimates <- do.call(rbind, estimation_out)
    
    # Save daily aggregated output
    out_df <- data.frame(
      loop = estimates$loop,
      T_PP = estimates$T_PP,
      ET = estimates$ET
    )
    
    daily <- aggregate(cbind(T_PP, ET) ~ loop, data=out_df, FUN=function(x) sum(x, na.rm=TRUE), na.action=na.pass)
    daily$T_ET_ratio <- daily$T_PP / daily$ET
    
    csv_file <- file.path(output_dir, paste0(site_name, "_PerezPriego_daily.csv"))
    write.csv(daily, csv_file, row.names = FALSE)
    cat(sprintf("  [Pérez-Priego] Saved: %s\n", csv_file))
    
    valid_ratio <- daily$T_ET_ratio[is.finite(daily$T_ET_ratio)]
    valid_ratio <- valid_ratio[valid_ratio >= 0 & valid_ratio <= 1]
    
    return(data.frame(
      site = site_name,
      method = "Perez-Priego",
      T_ET_mean = mean(valid_ratio, na.rm = TRUE),
      T_ET_median = median(valid_ratio, na.rm = TRUE),
      n_valid = length(valid_ratio),
      status = "OK"
    ))
    
  }, error = function(e) {
    cat(sprintf("  [Pérez-Priego] ERROR for %s: %s\n", site_name, e$message))
    return(data.frame(
      site = site_name,
      method = "Perez-Priego",
      T_ET_mean = NA,
      T_ET_median = NA,
      n_valid = 0,
      status = paste0("ERROR: ", e$message)
    ))
  })
}

# =============================================================================
# Main
# =============================================================================

args <- commandArgs(trailingOnly = TRUE)
preproc_files <- list.files(OUTPUT_DIR, pattern = "_preprocessed\\.nc$", full.names = TRUE)

if (length(args) > 0) {
  preproc_files <- preproc_files[grepl(paste(args, collapse = "|"), preproc_files)]
}

if (length(preproc_files) == 0) {
  cat("No preprocessed files found.\n")
  quit(status = 1)
}

all_results <- list()
for (i in seq_along(preproc_files)) {
  f <- preproc_files[i]
  site_name <- gsub("_preprocessed\\.nc$", "", basename(f))
  result <- run_pp_site(f, site_name, OUTPUT_DIR)
  all_results[[i]] <- result
}

summary_df <- do.call(rbind, all_results)
if (length(args) > 0) {
  # Avoid write collisions during parallel runs
  summary_file <- file.path(OUTPUT_DIR, paste0("perez_priego_summary_", paste(args, collapse = "_"), ".csv"))
} else {
  summary_file <- file.path(OUTPUT_DIR, "perez_priego_summary.csv")
}
write.csv(summary_df, summary_file, row.names = FALSE)
cat(sprintf("\nPérez-Priego summary saved: %s\n", summary_file))
print(summary_df)
