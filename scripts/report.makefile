ROOT = $(shell cd $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))/..; pwd)
SRC = $(wildcard *.py) $(shell find $(ROOT)/src/report_tools -name '*\.py')
MD = $(patsubst %.md.in,%.md,$(wildcard *.md.in))

.PHONY: all
all: docs

.PHONY: info
info:
	@echo ROOT: $(ROOT_DIR)
	@echo SRC: $(SRC)


%.md: %.md.in Makefile $(ROOT)/scripts/report.makefile $(SRC)
	codebraid pandoc \
		-f markdown -t markdown --no-cache --overwrite --standalone --self-contained \
		-o _tmp.md $<
	pandoc -f markdown \
		--to=markdown-smart-simple_tables-multiline_tables-grid_tables-fenced_code_attributes-inline_code_attributes-raw_attribute-pandoc_title_block-yaml_metadata_block \
		--toc -s _tmp.md -o $@

docs: $(MD)


