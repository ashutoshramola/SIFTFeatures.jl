#!/usr/bin/env julia
"""
Pure-Julia dataset downloader and SHA-256 verifier for Oxford (48 photos)
and optional HPatches (116 scenes / 696 photos). Uses only Julia stdlibs.

Usage:
    julia --project=. validation/download_datasets.jl --oxford validation-data
    julia --project=. validation/download_datasets.jl --hpatches hpatches-data
"""

using Downloads: Downloads
using SHA: sha256
using TOML: TOML

const HPATCHES_URL = "https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip"

file_sha256(path::AbstractString) = bytes2hex(open(sha256, path))

function verify_oxford_sequence(dir::AbstractString, files_sha::Dict{String,Any})
    isdir(dir) || return false
    for (fname, expected) in files_sha
        fpath = joinpath(dir, fname)
        isfile(fpath) || return false
        file_sha256(fpath) == expected || return false
    end
    return true
end

function ensure_oxford(data_dir::AbstractString;
                       manifest_path::AbstractString=joinpath(@__DIR__, "oxford-sources.toml"))
    manifest = TOML.parsefile(manifest_path)
    mkpath(data_dir)
    for seq in manifest["sequences"]
        name = seq["name"]
        seq_dir = joinpath(data_dir, name)
        files_sha = seq["files_sha256"]
        if verify_oxford_sequence(seq_dir, files_sha)
            println("oxford/$name: verified")
            continue
        end
        archive = joinpath(data_dir, "$name.tar.gz")
        if !isfile(archive) || file_sha256(archive) != seq["sha256"]
            println("oxford/$name: downloading $(seq["url"]) ...")
            tmp = archive * ".part"
            Downloads.download(seq["url"], tmp;
                               headers=["User-Agent" => "SIFTFeatures.jl/1.1 (+https://github.com/ashutoshramola/SIFTFeatures.jl)"])
            got = file_sha256(tmp)
            got == seq["sha256"] || error("oxford/$name archive SHA-256 mismatch: expected $(seq["sha256"]), got $got")
            mv(tmp, archive; force=true)
        end
        mkpath(seq_dir)
        run(`tar -xzf $archive -C $seq_dir`)
        verify_oxford_sequence(seq_dir, files_sha) ||
            error("oxford/$name extracted file checksum verification failed")
        println("oxford/$name: extracted and verified")
    end
    return data_dir
end

function find_hpatches_root(hpatches_dir::AbstractString)
    isdir(joinpath(hpatches_dir, "i_ajuntament")) && return String(hpatches_dir)
    nested = joinpath(hpatches_dir, "hpatches-sequences-release")
    isdir(joinpath(nested, "i_ajuntament")) && return nested
    return nothing
end

function ensure_hpatches(hpatches_dir::AbstractString)
    found = find_hpatches_root(hpatches_dir)
    !isnothing(found) && return found
    mkpath(hpatches_dir)
    archive = joinpath(hpatches_dir, "hpatches-sequences-release.zip")
    if !isfile(archive)
        println("hpatches: downloading $HPATCHES_URL ...")
        tmp = archive * ".part"
        Downloads.download(HPATCHES_URL, tmp)
        mv(tmp, archive; force=true)
    end
    println("hpatches: extracting $archive ...")
    if Sys.iswindows()
        run(`tar -xf $archive -C $hpatches_dir`)
    else
        run(`unzip -q -o $archive -d $hpatches_dir`)
    end
    found = find_hpatches_root(hpatches_dir)
    isnothing(found) && error("HPatches sequences not found under $hpatches_dir after extraction")
    return found
end

function _download_main(args)
    i = 1
    ran = false
    while i <= length(args)
        if args[i] == "--oxford"
            ensure_oxford(args[i + 1]); ran = true; i += 2
        elseif args[i] == "--hpatches"
            ensure_hpatches(args[i + 1]); ran = true; i += 2
        else
            error("Unknown argument: $(args[i])")
        end
    end
    ran || error("Pass --oxford <dir> and/or --hpatches <dir>")
end

if abspath(PROGRAM_FILE) == @__FILE__
    _download_main(ARGS)
end
