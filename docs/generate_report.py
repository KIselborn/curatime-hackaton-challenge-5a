#!/usr/bin/env python3
"""
Generate an educational PDF report explaining the GAI (Gut Aging Index) pipeline.

Usage:
    python generate_report.py

Output:
    GAI_Pipeline_Report.pdf

Dependencies:
    pip install fpdf2
"""

from fpdf import FPDF
from datetime import date


# ─── Custom PDF Class ─────────────────────────────────────────────────────────

class GAIReport(FPDF):
    """PDF report with consistent headers, footers, and helper methods."""

    MARGIN = 20
    BODY_FONT_SIZE = 11
    H1_SIZE = 20
    H2_SIZE = 15
    H3_SIZE = 13
    CODE_SIZE = 9
    LINE_H = 6  # default line height for body text

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "GAI Pipeline  |  Educational Guide", align="R")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── Section helpers ───────────────────────────────────────────────────

    def section_title(self, number, title):
        """Top-level section heading with a colored bar."""
        self.ln(4)
        self.set_fill_color(41, 98, 255)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", self.H1_SIZE)
        self.cell(0, 12, f"  {number}. {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        self.set_text_color(0, 0, 0)

    def sub_heading(self, title):
        """Second-level heading."""
        self.ln(2)
        self.set_font("Helvetica", "B", self.H2_SIZE)
        self.set_text_color(41, 98, 255)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def sub_sub_heading(self, title):
        """Third-level heading."""
        self.ln(1)
        self.set_font("Helvetica", "B", self.H3_SIZE)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        """Regular paragraph text with word wrapping."""
        self.set_font("Helvetica", "", self.BODY_FONT_SIZE)
        self.multi_cell(0, self.LINE_H, text)
        self.ln(2)

    def bold_text(self, text):
        """Bold paragraph."""
        self.set_font("Helvetica", "B", self.BODY_FONT_SIZE)
        self.multi_cell(0, self.LINE_H, text)
        self.ln(1)

    def italic_text(self, text):
        """Italic paragraph."""
        self.set_font("Helvetica", "I", self.BODY_FONT_SIZE)
        self.multi_cell(0, self.LINE_H, text)
        self.ln(1)

    def bullet(self, text):
        """Single bullet point."""
        self.set_font("Helvetica", "", self.BODY_FONT_SIZE)
        x = self.get_x()
        self.cell(6, self.LINE_H, "-")
        self.multi_cell(0, self.LINE_H, text)
        self.ln(1)

    def numbered_item(self, num, text):
        """Numbered list item."""
        self.set_font("Helvetica", "B", self.BODY_FONT_SIZE)
        self.cell(8, self.LINE_H, f"{num}.")
        self.set_font("Helvetica", "", self.BODY_FONT_SIZE)
        self.multi_cell(0, self.LINE_H, text)
        self.ln(1)

    def code_block(self, code):
        """Monospace code block with gray background."""
        self.ln(1)
        self.set_font("Courier", "", self.CODE_SIZE)
        self.set_fill_color(240, 240, 240)
        self.set_draw_color(200, 200, 200)

        lines = code.split("\n")
        # Calculate block height to check for page break
        line_h = 5
        block_h = line_h * len(lines) + 6
        if self.get_y() + block_h > self.h - 30:
            self.add_page()

        x_start = self.get_x()
        w = self.w - 2 * self.MARGIN

        # Draw background rectangle
        y_start = self.get_y()
        self.rect(x_start, y_start, w, block_h, style="FD")

        self.set_y(y_start + 3)
        for line in lines:
            self.set_x(x_start + 3)
            self.cell(w - 6, line_h, line)
            self.ln(line_h)
        self.ln(3)
        self.set_font("Helvetica", "", self.BODY_FONT_SIZE)

    def info_box(self, title, text):
        """Highlighted information box."""
        self.ln(2)
        self.set_fill_color(232, 240, 254)
        self.set_draw_color(41, 98, 255)
        self.set_font("Helvetica", "B", self.BODY_FONT_SIZE)

        w = self.w - 2 * self.MARGIN
        x = self.get_x()
        y = self.get_y()

        # Estimate height
        self.set_font("Helvetica", "", self.BODY_FONT_SIZE)
        # rough estimate: 1 line per 90 chars
        n_lines = max(1, len(text) // 80 + text.count("\n") + 1)
        box_h = 8 + n_lines * self.LINE_H + 4

        if y + box_h > self.h - 30:
            self.add_page()
            y = self.get_y()

        self.rect(x, y, w, box_h, style="FD")
        self.set_xy(x + 3, y + 2)
        self.set_font("Helvetica", "B", self.BODY_FONT_SIZE)
        self.cell(0, 7, title)
        self.set_xy(x + 3, y + 9)
        self.set_font("Helvetica", "", self.BODY_FONT_SIZE)
        self.multi_cell(w - 6, self.LINE_H, text)
        self.set_y(y + box_h + 2)

    def simple_table(self, headers, rows, col_widths=None):
        """Table with header row and alternating row colors."""
        w_avail = self.w - 2 * self.MARGIN
        n_cols = len(headers)
        if col_widths is None:
            col_widths = [w_avail / n_cols] * n_cols

        # Header
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(41, 98, 255)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)

        # Rows
        self.set_font("Helvetica", "", 9)
        for idx, row in enumerate(rows):
            if idx % 2 == 0:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1, fill=True)
            self.ln()
        self.ln(3)

    def wrapped_table(self, headers, rows, col_widths=None):
        """Table supporting multi-line cells via multi_cell."""
        w_avail = self.w - 2 * self.MARGIN
        n_cols = len(headers)
        if col_widths is None:
            col_widths = [w_avail / n_cols] * n_cols
        line_h = 5.5

        # Header
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(41, 98, 255)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()
        self.set_text_color(0, 0, 0)

        self.set_font("Helvetica", "", 9)
        for idx, row in enumerate(rows):
            if idx % 2 == 0:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)

            # Calculate row height based on longest cell
            max_lines = 1
            for i, val in enumerate(row):
                n = max(1, int(len(str(val)) * 2.2 / col_widths[i]) + 1)
                if n > max_lines:
                    max_lines = n
            row_h = line_h * max_lines

            # Check page break
            if self.get_y() + row_h > self.h - 25:
                self.add_page()

            y_start = self.get_y()
            x_start = self.get_x()
            for i, val in enumerate(row):
                x = x_start + sum(col_widths[:i])
                self.set_xy(x, y_start)
                self.cell(col_widths[i], row_h, "", border=1, fill=True)
                self.set_xy(x + 1, y_start + 1)
                self.multi_cell(col_widths[i] - 2, line_h, str(val))
            self.set_y(y_start + row_h)
        self.ln(3)


# ─── Content Sections ─────────────────────────────────────────────────────────

def add_title_page(pdf: GAIReport):
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(41, 98, 255)
    pdf.multi_cell(0, 14, "Gut Aging Index (GAI)\nPipeline Guide", align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "", 14)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "An Educational Walkthrough for Students", align="C",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(12)

    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 7, (
        "Based on: Bao Z. et al. (2024)\n"
        '"Gut Aging Index (GAI): A novel metric for assessing gut aging\n'
        'based on gut microbiota"\n'
        "Scientific Reports, 14, 29413"
    ), align="C")
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 8, f"Generated on {date.today().strftime('%B %d, %Y')}",
             align="C", new_x="LMARGIN", new_y="NEXT")


def add_introduction(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(1, "Introduction")

    pdf.body_text(
        "The Gut Aging Index (GAI) is a computational metric that estimates "
        "how much a person's gut microbiome has aged relative to their "
        "chronological (calendar) age. The central idea is simple: train a "
        "machine learning model to predict a person's age from the microbial "
        "species living in their gut, then compare the predicted age with the "
        "real age."
    )

    pdf.body_text(
        "If your gut microbiome looks older than your actual age (positive GAI), "
        "it may indicate an unhealthy shift. If it looks younger (negative GAI), "
        "it suggests a healthier-than-average microbial profile. This concept "
        "parallels similar aging clocks in genetics (epigenetic clocks) and "
        "brain imaging."
    )

    pdf.sub_heading("Why the Gut Microbiome?")
    pdf.body_text(
        "The human gut hosts trillions of microorganisms -- bacteria, archaea, "
        "fungi, and viruses -- collectively called the gut microbiome. Research "
        "has shown that the composition of these communities changes with age "
        "and is linked to conditions such as obesity, diabetes, inflammatory "
        "bowel disease, and even neurological disorders. By quantifying how "
        "the microbiome deviates from a healthy aging trajectory, GAI provides "
        "a single, interpretable number that summarizes gut health."
    )

    pdf.sub_heading("The Research Question")
    pdf.body_text(
        "Can we build a reliable, age-adjusted index from gut microbiome data "
        "that distinguishes healthy from unhealthy aging? The paper answers this "
        "by: (1) training a regression model exclusively on healthy individuals, "
        "(2) predicting microbiome-based age for all samples, and (3) applying "
        "an age-bin correction so the index is fair across different age groups."
    )

    pdf.sub_heading("Pipeline Overview")
    pdf.body_text(
        "The full pipeline has three stages that we will cover in this guide:"
    )
    pdf.numbered_item(1,
        "Data acquisition: Downloading raw microbiome data from the Qiita "
        "database using redbiom (download_data.sh).")
    pdf.numbered_item(2,
        "Data preparation: Converting raw BIOM files and supplementary "
        "metadata into the TSV format the model expects (prepare_data.py).")
    pdf.numbered_item(3,
        "Model training and GAI calculation: Training a regression model "
        "on healthy samples, predicting ages, and computing the corrected "
        "GAI score (gai_cal.py).")


def add_key_concepts(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(2, "Key Concepts Explained")

    pdf.sub_heading("OTUs (Operational Taxonomic Units)")
    pdf.body_text(
        "An OTU is a cluster of similar DNA sequences used as a proxy for a "
        "microbial species. When scientists sequence the 16S rRNA gene from a "
        "stool sample, they get millions of short DNA reads. These reads are "
        "grouped into OTUs based on sequence similarity (typically 97%). Each "
        "OTU roughly represents one bacterial species, and the number of reads "
        "in each OTU indicates its abundance in the sample."
    )

    pdf.sub_heading("16S rRNA Gene Sequencing")
    pdf.body_text(
        "The 16S rRNA gene is found in all bacteria and archaea. It contains "
        "both conserved regions (used to design universal primers) and variable "
        "regions (V1-V9) that differ between species. By sequencing the V4 "
        "region (as done in the AGP and GGMP studies), researchers can identify "
        "which bacteria are present and in what proportions. Both datasets in "
        "this project use Illumina 16S V4 150-nucleotide sequencing with "
        "closed-reference OTU picking against the Greengenes database."
    )

    pdf.sub_heading("BIOM Format")
    pdf.body_text(
        "BIOM (Biological Observation Matrix) is a standard file format for "
        "storing OTU tables. It is essentially a large matrix where rows are "
        "OTUs and columns are samples. The values represent how many sequence "
        "reads were assigned to each OTU in each sample. BIOM files use HDF5 "
        "binary format for efficient storage of sparse matrices (most entries "
        "are zero because each sample only contains a subset of all known OTUs)."
    )

    pdf.sub_heading("PyCaret")
    pdf.body_text(
        "PyCaret is a low-code Python library that automates machine learning "
        "workflows. Instead of manually importing, configuring, and comparing "
        "dozens of models, PyCaret provides functions like compare_models() that "
        "train and evaluate many algorithms in a single call. In this project, "
        "PyCaret version 2.3.5 is used for regression (predicting a continuous "
        "value -- age -- from OTU features)."
    )

    pdf.sub_heading("Regression in Machine Learning")
    pdf.body_text(
        "Regression is the task of predicting a continuous numeric value. "
        "Here, the input features are OTU abundances (how much of each "
        "bacterial species is present), and the target variable is the "
        "person's chronological age. The trained model learns patterns like "
        "'higher abundance of species X and lower abundance of species Y "
        "is associated with older age.' Common regression algorithms used "
        "include CatBoost, LightGBM, Random Forest, and Ridge Regression."
    )

    pdf.sub_heading("Health Stratification")
    pdf.body_text(
        "A critical design choice in this pipeline is that the regression "
        "model is trained only on healthy individuals. This ensures the model "
        "learns the 'normal' aging trajectory of gut microbiomes. When the "
        "model is then applied to unhealthy individuals, any deviation from "
        "the predicted age reflects the impact of disease on gut aging, not "
        "just normal variation."
    )

    pdf.sub_heading("Age-Bias Correction")
    pdf.body_text(
        "Regression models tend to predict ages closer to the mean of the "
        "training data. This means young people's predicted ages tend to be "
        "overestimated and old people's tend to be underestimated -- a known "
        "phenomenon called regression to the mean. The GAI pipeline corrects "
        "for this by computing the average GAI error in each age bin among "
        "healthy individuals and subtracting it. After correction, a GAI of "
        "zero for healthy individuals means their gut is aging as expected."
    )


def add_datasets(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(3, "Datasets")

    pdf.sub_heading("GGMP (Guangdong Gut Microbiome Project)")
    pdf.body_text(
        "The GGMP is a large-scale microbiome study from Guangdong Province, "
        "China (Qiita study ID 11757). It collected fecal samples and detailed "
        "health questionnaires from thousands of participants. After filtering "
        "for complete phenotypic information, the dataset contains approximately "
        "6,014 samples, of which 1,133 are classified as healthy."
    )

    pdf.sub_heading("AGP (American Gut Project)")
    pdf.body_text(
        "The AGP is one of the largest crowd-sourced microbiome studies "
        "(Qiita study ID 10317), with participants predominantly from the "
        "United States. After applying health filters, the dataset contains "
        "approximately 5,966 samples, of which 1,852 are classified as healthy."
    )

    pdf.sub_heading("Raw File Inventory")
    pdf.simple_table(
        ["File", "Description", "Source"],
        [
            ["GGMP-feces.biom", "OTU table (HDF5/BIOM)", "redbiom / Qiita"],
            ["AGP-feces.biom", "OTU table (HDF5/BIOM)", "redbiom / Qiita"],
            ["41598_..._MOESM2_ESM.xlsx", "Supplementary metadata", "Paper SI"],
        ],
        col_widths=[55, 65, 50],
    )

    pdf.sub_heading("Supplementary Excel Sheets")
    pdf.simple_table(
        ["Sheet", "Dataset", "Contents"],
        [
            ["Sup Table 4", "GGMP", "7,009 samples with ~40 disease columns"],
            ["Sup Table 6", "AGP", "5,966 samples with pre-computed health"],
        ],
        col_widths=[40, 30, 100],
    )

    pdf.sub_heading("Key Metadata Columns")
    pdf.wrapped_table(
        ["Column", "Type", "Description"],
        [
            ["id", "String", "Unique sample identifier (index)"],
            ["age", "Integer", "Chronological age of participant (>= 18)"],
            ["health", "String", "'y' = healthy, 'n' = non-healthy"],
            ["anthrop_BMI", "Float", "Body mass index (GGMP only)"],
            ["biochem_FBG", "Float", "Fasting blood glucose (GGMP only)"],
            ["antibiotics", "String", "'y'/'n' antibiotic use (GGMP only)"],
        ],
        col_widths=[40, 25, 105],
    )


def add_health_criteria(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(4, "Health Criteria")

    pdf.body_text(
        "Defining who counts as 'healthy' is critical because the regression "
        "model trains exclusively on healthy individuals. The two datasets use "
        "different criteria, reflecting the different questionnaires used in "
        "each study."
    )

    pdf.sub_heading("GGMP Health Definition (~40 conditions)")
    pdf.body_text(
        "A GGMP participant is classified as healthy only if ALL of the "
        "following are true:"
    )
    pdf.bullet("All ~38 binary disease columns (y/n) are 'n' (no disease)")
    pdf.bullet("malignant_tumor_disease is 'a' (absent)")
    pdf.bullet("MetS (metabolic syndrome) is 'n'")
    pdf.bullet("Fasting blood glucose (FBG) < 6.1 mmol/L")
    pdf.bullet("BMI < 24")
    pdf.bullet("No antibiotic use (antibiotics = 'n')")
    pdf.bullet("Complete phenotypic information for all fields")

    pdf.ln(2)
    pdf.sub_sub_heading("GGMP Disease Columns by Category")

    pdf.bold_text("Heart and Stroke (7 columns):")
    pdf.body_text(
        "heart_bypass_surgery, heart_stent_surgery, heart_angina_pectoris, "
        "heart_aspirin, heart_statins, stroke_ischemic, stroke_hemorrhagic"
    )

    pdf.bold_text("Respiratory (2 columns):")
    pdf.body_text("copd, asthma")

    pdf.bold_text("General Disease Systems (4 columns):")
    pdf.body_text(
        "osteoarticular_disease, waist_neck_disease, "
        "digestive_system_disease, urinary_system_disease"
    )

    pdf.bold_text("Specific Diseases (24 columns):")
    pdf.body_text(
        "dis_T1DM, dis_T2DM, dis_fatty_liver, dis_psoriasis, dis_AD, "
        "dis_PD, dis_ASD, dis_MS, dis_atherosclerosis, dis_LE, dis_ARDS, "
        "dis_gastritis, dis_hepatic_calculus, dis_cholecystitis, dis_colitis, "
        "dis_IBS, dis_kidneyStone, dis_gout, dis_AS, dis_RA, dis_neurosis, "
        "dis_CFS, dis_constipation_symptom, dis_diarrhea_symptom"
    )

    pdf.bold_text("Metabolic (1 column):")
    pdf.body_text("MetS (metabolic syndrome)")

    pdf.ln(2)
    pdf.sub_heading("AGP Health Definition (12 conditions)")
    pdf.body_text(
        "The AGP authors pre-computed a health column in Supplementary Table S6. "
        "A participant is healthy if they report 'I do not have this condition' "
        "for all 12 diseases below, AND BMI < 24, AND no antibiotic use in the "
        "past year."
    )
    pdf.simple_table(
        ["#", "Disease Column"],
        [
            ["1", "ibd (Inflammatory Bowel Disease)"],
            ["2", "alzheimers"],
            ["3", "asd (Autism Spectrum Disorder)"],
            ["4", "autoimmune"],
            ["5", "cancer"],
            ["6", "cardiovascular_disease"],
            ["7", "diabetes"],
            ["8", "ibs (Irritable Bowel Syndrome)"],
            ["9", "kidney_disease"],
            ["10", "liver_disease"],
            ["11", "lung_disease"],
            ["12", "mental_illness"],
        ],
        col_widths=[15, 155],
    )

    pdf.sub_heading("Cohort Summary After Filtering")
    pdf.simple_table(
        ["Cohort", "Total", "Healthy", "Non-Healthy", "Healthy %"],
        [
            ["GGMP", "6,014", "1,133", "4,881", "18.8%"],
            ["AGP", "5,966", "1,852", "4,114", "31.1%"],
        ],
        col_widths=[30, 30, 30, 40, 40],
    )


def add_data_processing(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(5, "Data Processing Pipeline")

    pdf.body_text(
        "This section walks through the two scripts that transform raw data "
        "into analysis-ready TSV files: download_data.sh and prepare_data.py."
    )

    pdf.sub_heading("Stage A: Downloading Raw Data (download_data.sh)")
    pdf.body_text(
        "The download script uses redbiom, a command-line tool for querying "
        "the Qiita microbial study database. For each dataset (GGMP and AGP), "
        "it performs three steps:"
    )
    pdf.numbered_item(1,
        "Search: Find all fecal sample IDs for the given study, excluding "
        "blanks/controls.")
    pdf.numbered_item(2,
        "Fetch metadata: Download sample metadata (age, health info, etc.) "
        "as a TSV file.")
    pdf.numbered_item(3,
        "Fetch samples: Download the OTU abundance table in BIOM format.")

    pdf.body_text(
        "The Qiita context used is 'Pick_closed-reference_OTUs-Greengenes-"
        "Illumina-16S-V4-150nt-bd7d4d', which specifies V4 16S sequencing "
        "with closed-reference OTU picking against the Greengenes database."
    )
    pdf.code_block(
        "# Example: download GGMP fecal samples\n"
        "redbiom search metadata \\\n"
        '  "feces where qiita_study_id == 11757" \\\n'
        "  | grep -vi blank > GGMP-samples-feces.txt\n"
        "\n"
        "redbiom fetch samples \\\n"
        "  --from GGMP-samples-feces.txt \\\n"
        "  --context Pick_closed-reference_OTUs-...-bd7d4d \\\n"
        "  --output GGMP-feces.biom"
    )

    pdf.sub_heading("Stage B: Data Preparation (prepare_data.py)")
    pdf.body_text(
        "The preparation script reads the downloaded BIOM files and "
        "supplementary metadata Excel file, applies health criteria, "
        "filters OTUs, and produces the meta.tsv and otu.tsv files "
        "required by gai_cal.py."
    )

    pdf.sub_sub_heading("Step 1: Load Supplementary Metadata")
    pdf.body_text(
        "The script reads the supplementary Excel file "
        "(41598_2024_82418_MOESM2_ESM.xlsx). For GGMP it uses 'Sup Table 4' "
        "(~7,009 samples with ~40 disease columns). For AGP it uses "
        "'Sup Table 6' (~5,966 samples with a pre-computed health column)."
    )

    pdf.sub_sub_heading("Step 2: Filter for Complete Data")
    pdf.body_text(
        "Samples missing critical fields (age, BMI, FBG, antibiotics, or "
        "disease columns) are removed. Only participants aged 18 or older "
        "are kept. This ensures the health classification is reliable."
    )

    pdf.sub_sub_heading("Step 3: Define Health Status")
    pdf.body_text(
        "For GGMP, health is computed from the ~40 disease columns plus "
        "biochemical thresholds (see Section 4). For AGP, the pre-computed "
        "'health' column from the authors is used directly."
    )

    pdf.sub_sub_heading("Step 4: Load and Filter BIOM Data")
    pdf.body_text(
        "The BIOM file is loaded and filtered to keep only samples present "
        "in the metadata. Then OTUs that appear in fewer than 10% of the "
        "remaining samples are removed. This prevalence filter eliminates "
        "rare OTUs that add noise without providing reliable signal."
    )
    pdf.code_block(
        "OTU_PREVALENCE_THRESHOLD = 0.10  # 10%\n"
        "# Keep OTUs present in >= 10% of filtered samples\n"
        "prevalence = (sparse_mat > 0).sum(axis=1) / n_samples\n"
        "keep_mask = prevalence >= prevalence_threshold"
    )

    pdf.sub_sub_heading("Step 5: ID Mapping (GGMP Only)")
    pdf.body_text(
        "GGMP sample IDs in the supplementary table (e.g. 'G440205594') "
        "must be mapped to the BIOM file format: '11757.G440205594.56280'. "
        "The script constructs this mapping and aligns the metadata with "
        "the OTU data. AGP sample IDs match directly."
    )

    pdf.sub_sub_heading("Step 6: Save Output TSVs")
    pdf.body_text(
        "Two TSV files are written for each dataset:"
    )
    pdf.bullet(
        "meta.tsv: Columns [id, age, health]. The 'id' column is the "
        "index. 'age' is an integer. 'health' is 'y' or 'n'."
    )
    pdf.bullet(
        "otu.tsv: Columns [id, OTU_1, OTU_2, ...]. The 'id' column is "
        "the index. Each OTU column contains integer abundance counts."
    )
    pdf.code_block(
        "# Output structure\n"
        "Processed-Data/\n"
        "  GGMP/\n"
        "    meta.tsv    # ~6,014 samples x 2 columns\n"
        "    otu.tsv     # ~6,014 samples x ~942 OTUs\n"
        "  AGP/\n"
        "    meta.tsv    # ~5,966 samples x 2 columns\n"
        "    otu.tsv     # ~5,966 samples x varies OTUs"
    )


def add_ml_pipeline(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(6, "Machine Learning Pipeline")

    pdf.body_text(
        "The core of the GAI calculation happens in gai_cal.py. This script "
        "takes the prepared meta.tsv and otu.tsv files, trains a regression "
        "model on healthy samples, predicts physiological age for everyone, "
        "and computes the age-corrected GAI score."
    )

    pdf.sub_heading("Step 1: Split by Health Status")
    pdf.bold_text("Function: split_otu_by_health(meta_path, otu_path)")
    pdf.body_text(
        "Reads both TSV files, joins them on the 'id' index, and creates "
        "three subsets: (a) healthy OTU data, (b) non-healthy OTU data, and "
        "(c) healthy OTU data merged with the 'age' column (the training "
        "dataset). The training data has OTU abundances as features and "
        "chronological age as the target."
    )
    pdf.code_block(
        "# Training data = healthy OTUs + age labels\n"
        "predicted_age_df = pd.merge(\n"
        "    healthy_otu_df,\n"
        '    meta_df["age"],\n'
        "    left_index=True, right_index=True,\n"
        '    how="inner"\n'
        ")"
    )

    pdf.sub_heading("Step 2: Model Training and Prediction")
    pdf.bold_text("Function: model_health_ages(predicted_age_df, otu_df)")
    pdf.body_text(
        "This function runs the full PyCaret automated ML workflow. "
        "It proceeds through four sub-steps:"
    )

    pdf.sub_sub_heading("2a. Setup")
    pdf.body_text(
        "PyCaret's setup() initializes the ML experiment. It receives the "
        "training data (healthy samples with OTU features and age target), "
        "sets a random seed (session_id=123) for reproducibility, and runs "
        "in silent mode (no interactive prompts). Internally, PyCaret splits "
        "the data into train/test folds, applies preprocessing, and "
        "configures the evaluation framework."
    )
    pdf.code_block(
        "reg = setup(\n"
        "    data=predicted_age_df,\n"
        '    target="age",\n'
        "    session_id=123,\n"
        "    silent=True\n"
        ")")

    pdf.sub_sub_heading("2b. Compare Models")
    pdf.body_text(
        "compare_models() trains and evaluates many regression algorithms "
        "(e.g. CatBoost, LightGBM, Random Forest, Ridge, Lasso, SVR, KNN, "
        "and more) using cross-validation. It ranks them by a default metric "
        "(R-squared) and returns the best-performing model. The comparison "
        "results are saved to compare_models.tsv."
    )

    pdf.sub_sub_heading("2c. Tune Model")
    pdf.body_text(
        "tune_model() performs hyperparameter optimization on the best model "
        "using random grid search with cross-validation. It tries different "
        "combinations of the model's hyperparameters (e.g. learning rate, "
        "tree depth, number of estimators) and selects the configuration "
        "that gives the best cross-validated performance. Results are saved "
        "to tuned_best_model.tsv."
    )

    pdf.sub_sub_heading("2d. Finalize and Predict")
    pdf.body_text(
        "finalize_model() retrains the tuned model on the entire training "
        "set (all healthy samples, no holdout). Then predict_model() applies "
        "this final model to ALL samples (healthy and non-healthy alike), "
        "generating predicted ages. The model is saved as a .pkl file with "
        "the current date."
    )
    pdf.code_block(
        "final_best_model = finalize_model(tuned_best_model)\n"
        "age_predictions = predict_model(\n"
        "    final_best_model, data=otu_df\n"
        ")\n"
        "# age_predictions['Label'] contains predicted ages\n"
        'save_model(final_best_model, f"final_best_model_{date}")')

    pdf.info_box(
        "Pre-trained Models",
        "The repository includes two pre-trained models in models/: "
        "agp_final_catboost.pkl (CatBoost, trained on AGP data) and "
        "ggmp_final_lightgbm.pkl (LightGBM, trained on GGMP data). "
        "These can be loaded with pycaret.regression.load_model() and "
        "used with predict_model() to skip the training step."
    )

    pdf.sub_heading("Step 3: Calculate Raw GAI")
    pdf.bold_text("Function: calculate_raw_gai(meta_df, age_predictions)")
    pdf.body_text(
        "The raw GAI is simply: predicted age minus chronological age."
    )
    pdf.code_block(
        "raw_GAI = predicted_age - chronological_age\n"
        '# In code: meta_df["raw GAI"] = \n'
        '#   age_predictions["Label"] - meta_df["age"]'
    )
    pdf.body_text(
        "A positive raw GAI means the model thinks your gut looks older "
        "than you are. A negative raw GAI means your gut looks younger."
    )

    pdf.sub_heading("Step 4: Calculate Age-Bin Adjust Values")
    pdf.bold_text("Function: calculate_adjust_value(meta_df)")
    pdf.body_text(
        "Because of regression to the mean, raw GAI is systematically biased: "
        "young healthy people tend to have positive raw GAI and old healthy "
        "people tend to have negative raw GAI. To correct this, the function "
        "groups healthy individuals into age bins and computes the mean raw "
        "GAI within each bin."
    )

    pdf.simple_table(
        ["Age Bin", "Range"],
        [
            ["1", "18-20"], ["2", "20-25"], ["3", "25-30"],
            ["4", "30-35"], ["5", "35-40"], ["6", "45-50"],
            ["7", "50-55"], ["8", "55-60"], ["9", "60-65"],
            ["10", "65-70"], ["11", "70-75"], ["12", "75-100"],
        ],
        col_widths=[30, 140],
    )

    pdf.info_box(
        "Note: Missing Age Bin 40-45",
        "The age ranges jump from (35,40) to (45,50), skipping the 40-45 "
        "range. Samples aged 40-44 will not receive an adjust value and "
        "will have NaN in their corrected GAI. This appears to be a bug in "
        "the original code."
    )

    pdf.body_text(
        "The mean raw GAI per bin is saved to adjust_values.tsv. Each "
        "sample is then assigned the adjust value corresponding to its "
        "age bin."
    )

    pdf.sub_heading("Step 5: Calculate Corrected GAI")
    pdf.bold_text("Function: calculate_corrected_gai(meta_df)")
    pdf.body_text(
        "The final corrected GAI subtracts the age-bin adjust value from "
        "the raw GAI:"
    )
    pdf.code_block(
        "corrected_GAI = raw_GAI - adjust_value"
    )
    pdf.body_text(
        "After this correction, the average corrected GAI for healthy "
        "individuals in each age bin should be close to zero. Positive "
        "values indicate accelerated gut aging; negative values indicate "
        "decelerated gut aging."
    )

    pdf.sub_heading("Step 6: Save Results")
    pdf.bold_text("Function: save_result(meta_df, result_path)")
    pdf.body_text(
        "The final DataFrame is written to result.tsv with columns: "
        "id (index), age, health, raw GAI, adjust value, corrected GAI."
    )


def add_output_files(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(7, "Output Files")

    pdf.body_text(
        "Running the full pipeline produces several output files. Below is "
        "a description of each."
    )

    pdf.wrapped_table(
        ["File", "Created By", "Contents"],
        [
            ["compare_models.tsv", "gai_cal.py",
             "Cross-validation results for all regression algorithms. Columns "
             "include MAE, MSE, RMSE, R2, etc. for each model."],
            ["tuned_best_model.tsv", "gai_cal.py",
             "Performance metrics after hyperparameter tuning of the best "
             "model from the comparison step."],
            ["final_best_model_YYYYMMDD.pkl", "gai_cal.py",
             "Serialized (pickled) final model trained on all healthy samples. "
             "Date-stamped. Load with pycaret load_model()."],
            ["adjust_values.tsv", "gai_cal.py",
             "Mean raw GAI per age bin for healthy individuals. Two columns: "
             "age_range and adjust_value."],
            ["result.tsv", "gai_cal.py",
             "Final output. Each row is a sample with columns: age, health, "
             "raw GAI, adjust value, corrected GAI."],
            ["Processed-Data/GGMP/meta.tsv", "prepare_data.py",
             "GGMP metadata: id (index), age, health (y/n)."],
            ["Processed-Data/GGMP/otu.tsv", "prepare_data.py",
             "GGMP OTU abundances after 10% prevalence filtering."],
            ["Processed-Data/AGP/meta.tsv", "prepare_data.py",
             "AGP metadata: id (index), age, health (y/n)."],
            ["Processed-Data/AGP/otu.tsv", "prepare_data.py",
             "AGP OTU abundances after 10% prevalence filtering."],
        ],
        col_widths=[55, 35, 80],
    )


def add_environment_setup(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(8, "Environment Setup")

    pdf.body_text(
        "This section provides step-by-step instructions for setting up a "
        "reproducible conda environment to run the entire pipeline."
    )

    pdf.sub_heading("Prerequisites")
    pdf.bullet("Anaconda or Miniconda installed on your system")
    pdf.bullet("Git for cloning the repository")
    pdf.bullet("Internet access for downloading packages and data")

    pdf.sub_heading("Step 1: Create a Conda Environment")
    pdf.body_text(
        "Create a new environment with Python 3.9 (compatible with "
        "PyCaret 2.3.5):"
    )
    pdf.code_block(
        "conda create -n gai python=3.9 -y\n"
        "conda activate gai"
    )

    pdf.sub_heading("Step 2: Install Core Dependencies")
    pdf.body_text(
        "Install the packages needed for the ML pipeline (gai_cal.py):"
    )
    pdf.code_block(
        "pip install pandas pycaret==2.3.5"
    )

    pdf.info_box(
        "PyCaret Version",
        "PyCaret 2.3.5 is pinned because the API changed significantly in "
        "version 3.x. Functions like setup(), compare_models(), and "
        "predict_model() have different parameter names and behaviors in "
        "newer versions. Do not upgrade without modifying the code."
    )

    pdf.sub_heading("Step 3: Install Data Preparation Dependencies")
    pdf.body_text(
        "If you want to run prepare_data.py to process raw BIOM files:"
    )
    pdf.code_block(
        "pip install numpy biom-format h5py scipy openpyxl"
    )

    pdf.sub_heading("Step 4: Install redbiom (Optional)")
    pdf.body_text(
        "If you want to download raw data from Qiita using download_data.sh:"
    )
    pdf.code_block(
        "pip install redbiom"
    )

    pdf.sub_heading("Step 5: Clone and Run")
    pdf.code_block(
        "git clone https://github.com/zwbao/GAI_paper.git\n"
        "cd GAI_paper\n"
        "\n"
        "# Option A: Run from raw data\n"
        "bash download_data.sh\n"
        "python prepare_data.py\n"
        "python gai_cal.py Processed-Data/GGMP/meta.tsv \\\n"
        "                  Processed-Data/GGMP/otu.tsv\n"
        "\n"
        "# Option B: Use pre-processed data directly\n"
        "python gai_cal.py meta.tsv otu.tsv"
    )

    pdf.sub_heading("Complete Environment Summary")
    pdf.simple_table(
        ["Package", "Version", "Purpose"],
        [
            ["Python", "3.9.x", "Runtime"],
            ["pandas", "latest", "Data manipulation"],
            ["pycaret", "2.3.5", "AutoML pipeline"],
            ["numpy", "latest", "Numerical operations"],
            ["biom-format", "latest", "Read BIOM OTU tables"],
            ["h5py", "latest", "HDF5 file support for BIOM"],
            ["scipy", "latest", "Sparse matrix support"],
            ["openpyxl", "latest", "Read Excel metadata"],
            ["redbiom", "latest", "Download from Qiita (optional)"],
        ],
        col_widths=[40, 30, 100],
    )


def add_verification(pdf: GAIReport):
    pdf.add_page()
    pdf.section_title(9, "Verification")

    pdf.body_text(
        "After running the pipeline, you can verify your results match "
        "the paper's reported values using these checks."
    )

    pdf.sub_heading("Check 1: Sample Counts")
    pdf.body_text("Compare your output sample counts with the paper:")
    pdf.simple_table(
        ["Dataset", "Total", "Healthy", "Non-Healthy"],
        [
            ["GGMP", "~6,014", "~1,133", "~4,881"],
            ["AGP", "~5,966", "~1,852", "~4,114"],
        ],
        col_widths=[40, 40, 40, 50],
    )
    pdf.code_block(
        "import pandas as pd\n"
        'meta = pd.read_csv("Processed-Data/GGMP/meta.tsv",\n'
        '                   sep="\\t", index_col="id")\n'
        "print(f'Total: {len(meta)}')\n"
        "print(meta['health'].value_counts())"
    )

    pdf.sub_heading("Check 2: OTU Counts After Filtering")
    pdf.body_text(
        "The paper reports ~942 OTUs for GGMP after the 10% prevalence "
        "filter. Check the number of columns in your OTU file:"
    )
    pdf.code_block(
        'otu = pd.read_csv("Processed-Data/GGMP/otu.tsv",\n'
        '                  sep="\\t", index_col="id")\n'
        "print(f'OTUs: {otu.shape[1]}')\n"
        "# Expected: ~942 for GGMP"
    )

    pdf.sub_heading("Check 3: Age Distribution")
    pdf.body_text(
        "Verify mean ages match the paper's reported statistics:"
    )
    pdf.simple_table(
        ["Cohort", "Group", "Mean Age (paper)"],
        [
            ["GGMP", "Healthy", "45.97 +/- 16.38"],
            ["GGMP", "Non-Healthy", "54.05 +/- 14.01"],
            ["AGP", "Healthy", "45.43 +/- 14.91"],
            ["AGP", "Non-Healthy", "49.57 +/- 14.15"],
        ],
        col_widths=[35, 40, 95],
    )
    pdf.code_block(
        "healthy = meta[meta['health'] == 'y']\n"
        "print(f\"Healthy mean age: \"\n"
        "      f\"{healthy['age'].mean():.2f} +/- \"\n"
        "      f\"{healthy['age'].std():.2f}\")"
    )

    pdf.sub_heading("Check 4: Model Performance")
    pdf.body_text(
        "Open compare_models.tsv after running gai_cal.py and compare "
        "the best model's R2 and MAE with the paper's reported values. "
        "The paper reports CatBoost as best for AGP and LightGBM as "
        "best for GGMP."
    )

    pdf.sub_heading("Check 5: Corrected GAI Distribution")
    pdf.body_text(
        "After computing corrected GAI, the mean corrected GAI for "
        "healthy individuals should be close to zero (by design). "
        "Non-healthy individuals should show higher (more positive) "
        "corrected GAI values, indicating accelerated gut aging."
    )
    pdf.code_block(
        'result = pd.read_csv("result.tsv", sep="\\t",\n'
        '                     index_col="id")\n'
        "h = result[result['health'] == 'y']\n"
        "nh = result[result['health'] == 'n']\n"
        "print(f\"Healthy GAI: {h['corrected GAI'].mean():.2f}\")\n"
        "print(f\"Non-Healthy GAI: {nh['corrected GAI'].mean():.2f}\")"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    pdf = GAIReport(orientation="P", unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_left_margin(GAIReport.MARGIN)
    pdf.set_right_margin(GAIReport.MARGIN)

    # Build all sections
    add_title_page(pdf)
    add_introduction(pdf)
    add_key_concepts(pdf)
    add_datasets(pdf)
    add_health_criteria(pdf)
    add_data_processing(pdf)
    add_ml_pipeline(pdf)
    add_output_files(pdf)
    add_environment_setup(pdf)
    add_verification(pdf)

    output_path = "GAI_Pipeline_Report.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}  ({pdf.page_no()} pages)")


if __name__ == "__main__":
    main()
