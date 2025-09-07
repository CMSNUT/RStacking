#' Reading Multiple Data Formats
#'
#' This function reads various data file formats including CSV, XLSX, XLS, and RDS,
#' and can also load built-in datasets from R packages.
#'
#' @param file_path Character string, file path or name of built-in dataset
#' @param package Character string, optional parameter specifying the package containing built-in data
#' @param ... Additional arguments passed to underlying reading functions
#'
#' @return Returns the read data frame or other data object
#'
#' @details
#' For CSV files, uses \code{readr::read_csv()};
#' For Excel files (XLSX/XLS), uses \code{openxlsx::read.xlsx()};
#' For RDS files, uses \code{readRDS()};
#' For built-in datasets, uses \code{data()} to load.
#'
#' @examples
#' \dontrun{
#' # Read CSV file
#' df_csv <- reading_data("data.csv")
#'
#' # Read Excel file
#' df_excel <- reading_data("data.xlsx", sheet = 1)
#'
#' # Read RDS file
#' df_rds <- reading_data("data.rds")
#'
#' # Load built-in dataset
#' df_attrition <- reading_data("attrition", package = "modeldata")
#' }
#'
#' @author 
#' Donghui Xia \email{dhaxia@snut.edu.cn}
#' School of Chemical and Environmental Science, Shaanxi University of Technology
#' ORCID: 0000-0002-2664-7596
#'
#' @seealso
#' \code{\link[readr]{read_csv}}, \code{\link[openxlsx]{read.xlsx}}, \code{\link{readRDS}}
#'
#' @importFrom readr read_csv
#' @importFrom openxlsx read.xlsx
#' @importFrom tools file_ext
#' @export
reading_data <- function(file_path, package = NULL, ...) {
  # If package parameter is specified, try to load built-in data from package
  if (!is.null(package)) {
    # Check if package is installed
    if (!requireNamespace(package, quietly = TRUE)) {
      stop("Please install package: ", package, "\nUse command: install.packages('", package, "')")
    }
    
    # Check if dataset exists in package
    if (!file_path %in% data(package = package)$results[, "Item"]) {
      stop("Dataset '", file_path, "' does not exist in package '", package, "'")
    }
    
    # Load dataset
    data(list = file_path, package = package, envir = environment())
    return(get(file_path))
  }
  
  # Check if file exists
  if (!file.exists(file_path)) {
    stop("File does not exist: ", file_path)
  }
  
  # Extract file extension and convert to lowercase
  file_ext <- tolower(tools::file_ext(file_path))
  
  # Select reading method based on extension
  switch(file_ext,
         "csv" = {
           # Use readr package for CSV files
           if (!requireNamespace("readr", quietly = TRUE)) {
             stop("Please install readr package: install.packages('readr')")
           }
           return(readr::read_csv(file_path, ...))
         },
         "xlsx" = {
           # Use openxlsx package for XLSX files
           if (!requireNamespace("openxlsx", quietly = TRUE)) {
             stop("Please install openxlsx package: install.packages('openxlsx')")
           }
           return(openxlsx::read.xlsx(file_path, ...))
         },
         "xls" = {
           # Use openxlsx package for XLS files
           if (!requireNamespace("openxlsx", quietly = TRUE)) {
             stop("Please install openxlsx package: install.packages('openxlsx')")
           }
           return(openxlsx::read.xlsx(file_path, ...))
         },
         "rds" = {
           # Use base R for RDS files
           return(readRDS(file_path, ...))
         },
         {
           # Unsupported format
           stop("Unsupported file format: ", file_ext)
         }
  )
}