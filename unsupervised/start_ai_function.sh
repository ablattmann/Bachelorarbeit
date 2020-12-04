#!/bin/bash
# KI Absicherung
#
# AI function start script template

die()
{
	local _ret=$2
	test -n "$_ret" || _ret=1
	test "$_PRINT_HELP" = yes && print_help >&2
	echo "$1" >&2
	exit ${_ret}
}


begins_with_short_option()
{
	local first_option all_short_options='siomch'
	first_option="${1:0:1}"
	test "$all_short_options" = "${all_short_options/$first_option/}" && return 1 || return 0
}

# THE DEFAULTS INITIALIZATION - OPTIONALS
_arg_software="${HOME}/software"
_arg_input="${HOME}/input"
_arg_output="${HOME}/output"
_arg_mode="train"
_arg_config=""
_arg_parse_config=false
_arg_gpu=-1


print_help()
{
	printf '%s\n' "This script helps with the execution of the KI Absicherung AI algorithms."
	printf 'Usage: %s [-s|--software <arg>] [---i|--input <arg>] [-o|--output <arg>] [-m|--mode <arg>] [-c|--config <arg>] [--parse-config] [-h|--help]\n' "$0"
	printf '\t%s\n' "-s, --software: relative path to  directory where code has to be executed, (default: './software')"
	printf '\t%s\n' "-i, --input: relative path to input data folder (default: './input')"
	printf '\t%s\n' "-o, --output: relative path to output data folder (default: './output')"
	printf '\t%s\n' "-m, --mode: mode flag that indicates the modus how the ai function should start in, e.g. train, test  (default: 'train')"
	printf '\t%s\n' "-c, --config: config file passed to the ai function (default: '')"
	printf '\t%s\n' "--parse-config: if present then --config is expected to be a .json/.yaml file that is parsed into command line options"
	printf '\t%s\n' "-h, --help: Prints help"
	printf '%s\n\n' "Example: ai_function_template.sh --software ./software/ai_function --input ./input --output ./output --mode train --config ./config.json"
}


parse_commandline()
{
	_positionals_count=0
	while test $# -gt 0
	do
		_key="$1"
		case "$_key" in
			-s|--software)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_software="$2"
				shift
				;;
			--software=*)
				_arg_software="${_key##--software=}"
				;;
			-s*)
				_arg_software="${_key##-s}"
				;;
		  -g|--gpu)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_gpu="$2"
				shift
				;;
			--gpu=*)
				_arg_gpu="${_key##--gpu=}"
				;;
			-g*)
				_arg_gpu="${_key##-g}"
				;;
			-i|--input)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_input="$2"
				shift
				;;
			--input=*)
				_arg_input="${_key##--input=}"
				;;
			-i*)
				_arg_input="${_key##-i}"
				;;
			-o|--output)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_output="$2"
				shift
				;;
			--output=*)
				_arg_output="${_key##--output=}"
				;;
			-o*)
				_arg_output="${_key##-o}"
				;;
			-m|--mode)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_mode="$2"
				shift
				;;
			--mode=*)
				_arg_mode="${_key##--mode=}"
				;;
			-m*)
				_arg_mode="${_key##-m}"
				;;
			-c|--config)
				test $# -lt 2 && die "Missing value for the optional argument '$_key'." 1
				_arg_config="$2"
				shift
				;;
			--config=*)
				_arg_config="${_key##--config=}"
				;;
			-c*)
				_arg_config="${_key##-c}"
				;;
			-h|--help)
				print_help
				exit 0
				;;
			-h*)
				print_help
				exit 0
				;;
			*)
				_PRINT_HELP=yes die "FATAL ERROR: Got an unexpected argument '$1'" 1
				;;
		esac
		shift
	done
}

parse_commandline "$@"

# Strip leading whitespace
_arg_software="$(echo -e "${_arg_software}" | sed -e 's/^[[:space:]]*//')"
_arg_input="$(echo -e "${_arg_input}" | sed -e 's/^[[:space:]]*//')"
_arg_output="$(echo -e "${_arg_output}" | sed -e 's/^[[:space:]]*//')"
_arg_mode="$(echo -e "${_arg_mode}" | sed -e 's/^[[:space:]]*//')"
_arg_config="$(echo -e "${_arg_config}" | sed -e 's/^[[:space:]]*//')"
_arg_gpu="$(echo -e "${_arg_gpu}" | sed -e 's/^[[:space:]]*//')"


# Remove trailing options
software_dir=${_arg_software%% *}
input_dir=${_arg_input%% *}
output_dir=${_arg_output%% *}
mode_flag=${_arg_mode%% *}
config_file=${_arg_config%% *}
gpu_id=${_arg_gpu%% *}

echo ""
echo -n  "Check for software directory: "
test -d "$software_dir" || die "\"$software_dir\" does not exist. Exiting."
echo "$software_dir exists!"

echo -n "Check for input directory: "
test -d "$input_dir" || die "\"$input_dir\" does not exist. Exiting."
echo "$input_dir exists!"

echo -n "Check for output directory: "
test -d "$output_dir" || die "\"$output_dir\" does not exist. Exiting."
echo "$output_dir exists!"

echo -n "Config file is ${config_file}"
echo -n ""

echo -n "GPU id is ${gpu_id}"
echo -n ""

command=""

############################### Modify Here ##########################################

#### Add here the specific software call and the optional or positional options to the software call
software="main.py"

### Add here modifications to the input option, e.g. add a filename to the input directory path
input="${input_dir}"

### Add here modifications to the output option, e.g. add a filename to the output directory path
output="${output_dir}"

#### Add here modifications to the config option, e.g. add a "--config " option before the filename
#if ${parse_config}; then
#	# if --parse-config option is set then set config to the parsed options
config="${config_file}"
#else
#	config='--config ""'
#fi
gpu="${gpu_id}"

### Add software, input, output and optional config together for the final CLI command that is to be called
command="${software} --input ${input} --output ${output} --mode ${mode_flag} --config ${config} --gpu ${gpu}"

############################### End Modification ######################################

echo ""
echo "Start execution of \"$command\":"

if [[ ${command%% *} == *.sh ]]; then
	bash $command
elif [[ ${command%% *} == *.py ]]; then
	python $command
elif [ -z "$var" ]; then
	die "Command was not defined! Modify start.sh script to define a CLI command to call the ai functions. Exiting." 1
else
	die "${command%% *} is not yet supported (only *.sh, *.py). Exiting." 1
fi

echo "Done. The ouput is stored in $output"
