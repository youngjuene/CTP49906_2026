# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
# ]
# ///

import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    import json
    from pathlib import Path
    import sys
    from uuid import uuid4

    _repo_root = Path(__file__).resolve().parents[1]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from curriculum_common.audience_packets import AudiencePacket, AudienceReading
    from curriculum_common.json_import import parse_json_object
    from digital_storytelling.workflow import (
        ACCESSIBILITY_OVERLAY_POLICY,
        COMPARISON_ROUTE_ID,
        build_blinded_packet,
        build_private_story_bundle,
        content_for,
        create_story_artifact,
        replay_payload,
        require_complete_audience_exchange,
        serialize_private_story_bundle,
        utc_now_text,
        validate_replay,
    )

    session_pseudonym = f"story-{uuid4().hex[:12]}"
    get_artifacts, set_artifacts = mo.state(tuple())
    get_packet, set_packet = mo.state(None)
    get_readings, set_readings = mo.state(tuple())
    get_structure_note, set_structure_note = mo.state(None)
    get_disagreement_note, set_disagreement_note = mo.state(None)
    saved_replay = replay_payload()
    validate_replay(saved_replay)
    return (
        ACCESSIBILITY_OVERLAY_POLICY,
        AudiencePacket,
        AudienceReading,
        COMPARISON_ROUTE_ID,
        build_blinded_packet,
        build_private_story_bundle,
        content_for,
        create_story_artifact,
        get_artifacts,
        get_disagreement_note,
        get_packet,
        get_readings,
        get_structure_note,
        json,
        parse_json_object,
        require_complete_audience_exchange,
        saved_replay,
        serialize_private_story_bundle,
        session_pseudonym,
        set_artifacts,
        set_disagreement_note,
        set_packet,
        set_readings,
        set_structure_note,
        utc_now_text,
    )


@app.cell(hide_code=True)
def _(mo):
    language_control = mo.ui.dropdown(
        ["en", "ko"],
        value="en",
        label="Language / 언어 (en = English, ko = 한국어)",
    )
    language_control
    return (language_control,)


@app.cell
def _(content_for, language_control):
    language = language_control.value or "en"
    text = content_for(language)
    return language, text


@app.cell(hide_code=True)
def _(mo, text):
    mo.md(
        f"""
        # {text['title']}

        **{text['question']}**

        {text['route_note']}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, text):
    mo.hstack(
        [
            mo.stat(
                value=text["teaching_mode"],
                label="Operating mode / 운영 모드",
                caption=text["teaching_caption"],
                bordered=True,
            ),
            mo.stat(
                value=text["egress"],
                label="Student-data egress / 학생 자료 반출",
                caption=text["egress_caption"],
                bordered=True,
            ),
            mo.stat(
                value=text["release_status"],
                label="Fidelity status / 충실도 상태",
                caption=text["release_caption"],
                bordered=True,
            ),
        ],
        widths="equal",
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo, text):
    mo.md(
        f"""
        ## {text['section_prepare']}

        ### {text['stage_orient']}

        **Required / 필수.** {text['orient_body']}

        **Data boundary / 데이터 경계.** {text['boundary']}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, text):
    def _validate_pre(_value):
        if not _value:
            return text["pre_submit"]
        if any(not str(_value[_key]).strip() for _key in _value):
            return text["pre_submit"]
        return None

    pre_task_form = mo.md(
        f"""
        ### {text['pre_heading']}

        {text['pre_prompt']}

        **{text['pre_evidence_label']}** {{evidence}}

        **{text['pre_alternative_label']}** {{alternative}}

        **{text['pre_limit_label']}** {{limit}}
        """
    ).batch(
        evidence=mo.ui.text_area(
            label=text["pre_evidence_label"], rows=3, full_width=True
        ),
        alternative=mo.ui.text_area(
            label=text["pre_alternative_label"], rows=3, full_width=True
        ),
        limit=mo.ui.text_area(
            label=text["pre_limit_label"], rows=3, full_width=True
        ),
    ).form(
        submit_button_label=text["pre_submit"],
        validate=_validate_pre,
        bordered=True,
    )
    pre_task_form
    return (pre_task_form,)


@app.cell(hide_code=True)
def _(mo, text):
    mo.md(
        f"""
        ## {text['section_make']}

        ### {text['stage_plan']}

        **Required / 필수.** {text['plan_body']}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, text):
    def _validate_v1(_value):
        if not _value:
            return text["v1_waiting"]
        _file_value = _value["media"]
        if not _file_value or not _file_value[0].contents:
            return text["v1_upload"]
        _required_fields = tuple(_key for _key in _value if _key not in {"media", "reviewed"})
        if any(not str(_value[_key]).strip() for _key in _required_fields):
            return text["v1_waiting"]
        if _value["reviewed"] is not True:
            return text["overlay_review"]
        return None

    v1_form = mo.md(
        f"""
        **{text['v1_upload']}** {{media}}

        **{text['intention']}** {{intention}}

        **{text['audience']}** {{audience}}

        **{text['relation']}** {{relation}}

        **{text['tags']}** {{tags}}

        **{text['context']}** {{context}}

        **{text['source_alias']}** {{source_alias}}

        **{text['source_rights']}** {{source_rights}}

        **{text['editor']}** {{editor}}

        **{text['ordering']}** {{ordering}}

        **{text['trims']}** {{trims}}

        **{text['mix']}** {{mix}}

        **{text['caption']}** {{caption}}

        **{text['description']}** {{description}}

        {{reviewed}} **{text['overlay_review']}**

        **{text['assistance']}** {{assistance}}

        **{text['export_preset']}** {{export_preset}}

        **{text['change_rationale_v1']}** {{rationale}}
        """
    ).batch(
        media=mo.ui.file(label=text["v1_upload"], filetypes=["video/*"], multiple=False),
        intention=mo.ui.text_area(label=text["intention"], rows=2, full_width=True),
        audience=mo.ui.text(label=text["audience"], full_width=True),
        relation=mo.ui.text_area(label=text["relation"], rows=2, full_width=True),
        tags=mo.ui.text(label=text["tags"], full_width=True),
        context=mo.ui.text_area(label=text["context"], rows=2, full_width=True),
        source_alias=mo.ui.text(label=text["source_alias"], full_width=True),
        source_rights=mo.ui.text_area(label=text["source_rights"], rows=2, full_width=True),
        editor=mo.ui.text(label=text["editor"], full_width=True),
        ordering=mo.ui.text_area(label=text["ordering"], rows=2, full_width=True),
        trims=mo.ui.text_area(label=text["trims"], rows=2, full_width=True),
        mix=mo.ui.text_area(label=text["mix"], rows=2, full_width=True),
        caption=mo.ui.text_area(label=text["caption"], rows=3, full_width=True),
        description=mo.ui.text_area(label=text["description"], rows=3, full_width=True),
        reviewed=mo.ui.checkbox(label=text["overlay_review"], value=False),
        assistance=mo.ui.text_area(label=text["assistance"], rows=2, full_width=True),
        export_preset=mo.ui.text(label=text["export_preset"], full_width=True),
        rationale=mo.ui.text_area(label=text["change_rationale_v1"], rows=2, full_width=True),
    ).form(
        submit_button_label=text["v1_submit"],
        validate=_validate_v1,
        bordered=True,
    )
    v1_form
    return (v1_form,)


@app.cell(hide_code=True)
def _(
    create_story_artifact,
    get_artifacts,
    mo,
    set_artifacts,
    text,
    utc_now_text,
    v1_form,
):
    _v1_card = mo.callout(mo.md(text["v1_waiting"]), kind="info")
    _v1_value = v1_form.value
    if _v1_value is not None:
        _existing_artifacts = get_artifacts()
        try:
            _new_v1 = create_story_artifact(
                media_bytes=_v1_value["media"][0].contents,
                version_label="V1",
                parent_artifact_id=None,
                local_registered_at_utc=utc_now_text(),
                event_index=0,
                elapsed_ms=0,
                creator_intention=_v1_value["intention"],
                intended_audience=_v1_value["audience"],
                sound_image_relation=_v1_value["relation"],
                concept_tags=_v1_value["tags"],
                cultural_aesthetic_context=_v1_value["context"],
                source_alias=_v1_value["source_alias"],
                source_permission_note=_v1_value["source_rights"],
                editor_name_version=_v1_value["editor"],
                ordering=_v1_value["ordering"],
                trims=_v1_value["trims"],
                mix_levels=_v1_value["mix"],
                caption_or_transcript=_v1_value["caption"],
                visual_description=_v1_value["description"],
                overlays_human_reviewed=_v1_value["reviewed"],
                assistance_disclosure=_v1_value["assistance"],
                export_preset_version=_v1_value["export_preset"],
                change_rationale=_v1_value["rationale"],
            )
            _existing_v1 = next(
                (_item for _item in _existing_artifacts if _item.version_label == "V1"),
                None,
            )
            if _existing_v1 is None:
                set_artifacts((_new_v1,))
            elif _existing_v1.artifact_id != _new_v1.artifact_id:
                raise ValueError("V1 is immutable; reset the session before replacing it")
        except Exception as _error:  # noqa: BLE001 — unsafe student input belongs in UI
            _v1_card = mo.callout(
                mo.md(f"**Registration rejected** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            _v1_card = mo.callout(mo.md(text["v1_complete"]), kind="success")
    _v1_card
    return


@app.cell(hide_code=True)
def _(mo, text):
    mo.callout(mo.md(text["overlay_policy"]), kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo, text):
    mo.md(
        f"""
        ## {text['section_activity']}

        ### {text['stage_structure']}

        **Required / 필수.** {text['structure_body']}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, text):
    def _validate_structure(_value):
        if not _value or any(not str(_value[_key]).strip() for _key in _value):
            return text["structure_submit"]
        return None

    structure_form = mo.md(
        f"""
        **{text['turning_point']}** {{turning_point}}

        **{text['alternative_sequence']}** {{alternative_sequence}}

        **{text['sound_role']}** {{sound_role}}

        **{text['expected_reading']}** {{expected_reading}}
        """
    ).batch(
        turning_point=mo.ui.text_area(label=text["turning_point"], rows=2, full_width=True),
        alternative_sequence=mo.ui.text_area(
            label=text["alternative_sequence"], rows=2, full_width=True
        ),
        sound_role=mo.ui.text_area(label=text["sound_role"], rows=2, full_width=True),
        expected_reading=mo.ui.text_area(
            label=text["expected_reading"], rows=2, full_width=True
        ),
    ).form(
        submit_button_label=text["structure_submit"],
        validate=_validate_structure,
        bordered=True,
    )
    structure_form
    return (structure_form,)


@app.cell
def _(set_structure_note, structure_form):
    if structure_form.value is not None:
        set_structure_note(dict(structure_form.value))
    return


@app.cell(hide_code=True)
def _(mo, text):
    mo.md(
        f"""
        ### {text['stage_audience']}

        **Required / 필수.** {text['audience_body']}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, text):
    def _validate_packet(_value):
        if not _value:
            return text["packet_waiting"]
        for _key in ("asset_reference", "media_type", "duration_ms", "caption_reference"):
            if not str(_value[_key]).strip():
                return text["packet_waiting"]
        if _value["sharing_permission"] is not True:
            return text["sharing_permission"]
        return None

    packet_form = mo.md(
        f"""
        **{text['asset_reference']}** {{asset_reference}}

        **{text['media_type']}** {{media_type}}

        **{text['duration_ms']}** {{duration_ms}}

        **{text['caption_reference']}** {{caption_reference}}

        {{sharing_permission}} **{text['sharing_permission']}**
        """
    ).batch(
        asset_reference=mo.ui.text(label=text["asset_reference"], full_width=True),
        media_type=mo.ui.dropdown(
            {"video/mp4": "video/mp4", "video/webm": "video/webm"},
            value="video/mp4",
            label=text["media_type"],
        ),
        duration_ms=mo.ui.number(
            start=1, step=1, value=1, label=text["duration_ms"], full_width=True
        ),
        caption_reference=mo.ui.text(
            label=text["caption_reference"], full_width=True
        ),
        sharing_permission=mo.ui.checkbox(
            label=text["sharing_permission"], value=False
        ),
    ).form(
        submit_button_label=text["packet_submit"],
        validate=_validate_packet,
        bordered=True,
    )
    packet_form
    return (packet_form,)


@app.cell(hide_code=True)
def _(
    build_blinded_packet,
    get_artifacts,
    get_packet,
    json,
    language,
    mo,
    packet_form,
    set_packet,
    text,
):
    _packet_card = mo.callout(mo.md(text["packet_waiting"]), kind="info")
    _packet_value = packet_form.value
    if _packet_value is not None:
        try:
            _v1_artifact = next(
                _item for _item in get_artifacts() if _item.version_label == "V1"
            )
            _current_packet = get_packet()
            if _current_packet is None:
                _current_packet = build_blinded_packet(
                    exchange_artifact_id=f"exchange-{__import__('uuid').uuid4().hex}",
                    asset_reference=_packet_value["asset_reference"],
                    media_type=_packet_value["media_type"],
                    duration_ms=int(_packet_value["duration_ms"]),
                    caption_reference=_packet_value["caption_reference"],
                    accessibility_note=(
                        "Human-correctable caption/description overlays are available; "
                        "creator context remains excluded during blinded response."
                    ),
                    language=language,
                    permission_confirmed=_packet_value["sharing_permission"],
                )
                set_packet(_current_packet)
            _packet_bytes = json.dumps(
                _current_packet.to_dict(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _packet_card = mo.vstack(
                [
                    mo.callout(mo.md(text["packet_ready"]), kind="success"),
                    mo.download(
                        data=_packet_bytes,
                        filename=f"{_current_packet.exchange_artifact_id}.json",
                        label=text["packet_download"],
                    ),
                    mo.md(f"`{_v1_artifact.version_label}` · `{_current_packet.exchange_artifact_id}`"),
                ]
            )
        except Exception as _error:  # noqa: BLE001 — unsafe student input belongs in UI
            _packet_card = mo.callout(
                mo.md(f"**Packet rejected** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
    _packet_card
    return


@app.cell(hide_code=True)
def _(mo, text):
    def _validate_audience_import(_value):
        if not _value or not _value["packet"] or len(_value["readings"] or ()) < 2:
            return text["audience_import_submit"]
        return None

    audience_import_form = mo.md(
        f"""
        **{text['audience_packet_upload']}** {{packet}}

        **{text['audience_readings_upload']}** {{readings}}
        """
    ).batch(
        packet=mo.ui.file(
            label=text["audience_packet_upload"], filetypes=[".json"], multiple=False
        ),
        readings=mo.ui.file(
            label=text["audience_readings_upload"], filetypes=[".json"], multiple=True
        ),
    ).form(
        submit_button_label=text["audience_import_submit"],
        validate=_validate_audience_import,
        bordered=True,
    )
    audience_import_form
    return (audience_import_form,)


@app.cell(hide_code=True)
def _(
    AudiencePacket,
    AudienceReading,
    audience_import_form,
    mo,
    parse_json_object,
    require_complete_audience_exchange,
    set_packet,
    set_readings,
    text,
):
    _audience_card = mo.callout(mo.md(text["audience_waiting"]), kind="info")
    _audience_value = audience_import_form.value
    if _audience_value is not None:
        try:
            _imported_packet = AudiencePacket.from_mapping(
                parse_json_object(_audience_value["packet"][0].contents)
            )
            _imported_readings = tuple(
                AudienceReading.from_mapping(
                    parse_json_object(_file.contents), packet=_imported_packet
                )
                for _file in _audience_value["readings"]
            )
            require_complete_audience_exchange(_imported_packet, _imported_readings)
        except Exception as _error:  # noqa: BLE001 — unsafe imports belong in UI
            _audience_card = mo.callout(
                mo.md(f"**Audience import rejected** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            set_packet(
                _imported_packet.reveal(
                    protocol_deviation="creator context revealed after valid classroom import"
                )
            )
            set_readings(_imported_readings)
            _reading_lines = "\n".join(
                f"- `{_reading.audience_pseudonym}`: {_reading.open_interpretation}"
                for _reading in _imported_readings
            )
            _audience_card = mo.callout(
                mo.md(f"{text['audience_complete']}\n\n{_reading_lines}"),
                kind="success",
            )
    _audience_card
    return


@app.cell(hide_code=True)
def _(get_artifacts, get_readings, mo, text):
    _readings_ready = len(get_readings()) >= 2
    _v1_for_reveal = next(
        (_item for _item in get_artifacts() if _item.version_label == "V1"), None
    )
    if _readings_ready and _v1_for_reveal is not None:
        _creator_context_card = mo.callout(
            mo.md(
                f"**{text['intention']}** — {_v1_for_reveal.creator_intention}\n\n"
                f"**{text['relation']}** — {_v1_for_reveal.sound_image_relation}"
            ),
            kind="neutral",
        )
    else:
        _creator_context_card = mo.callout(mo.md(text["audience_waiting"]), kind="info")
    _creator_context_card
    return


@app.cell(hide_code=True)
def _(mo, text):
    def _validate_disagreement(_value):
        if not _value or any(not str(_value[_key]).strip() for _key in _value):
            return text["disagreement_submit"]
        return None

    disagreement_form = mo.md(
        f"""
        **{text['disagreement']}** {{pattern}}

        **{text['disagreement_context']}** {{context}}
        """
    ).batch(
        pattern=mo.ui.text_area(label=text["disagreement"], rows=3, full_width=True),
        context=mo.ui.text_area(
            label=text["disagreement_context"], rows=3, full_width=True
        ),
    ).form(
        submit_button_label=text["disagreement_submit"],
        validate=_validate_disagreement,
        bordered=True,
    )
    disagreement_form
    return (disagreement_form,)


@app.cell
def _(disagreement_form, get_readings, set_disagreement_note):
    if disagreement_form.value is not None and len(get_readings()) >= 2:
        set_disagreement_note(dict(disagreement_form.value))
    return


@app.cell(hide_code=True)
def _(mo, text):
    mo.md(
        f"""
        ### {text['stage_revise']}

        **Required / 필수.** {text['v2_waiting']}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, text):
    def _validate_v2(_value):
        if not _value or not _value["media"] or not _value["media"][0].contents:
            return text["v2_waiting"]
        for _key in ("change", "caption", "description"):
            if not str(_value[_key]).strip():
                return text["v2_waiting"]
        if _value["reviewed"] is not True:
            return text["overlay_review"]
        return None

    v2_form = mo.md(
        f"""
        **{text['v2_upload']}** {{media}}

        **{text['v2_change']}** {{change}}

        **{text['caption']}** {{caption}}

        **{text['description']}** {{description}}

        {{reviewed}} **{text['overlay_review']}**
        """
    ).batch(
        media=mo.ui.file(label=text["v2_upload"], filetypes=["video/*"], multiple=False),
        change=mo.ui.text_area(label=text["v2_change"], rows=3, full_width=True),
        caption=mo.ui.text_area(label=text["caption"], rows=3, full_width=True),
        description=mo.ui.text_area(label=text["description"], rows=3, full_width=True),
        reviewed=mo.ui.checkbox(label=text["overlay_review"], value=False),
    ).form(
        submit_button_label=text["v2_submit"],
        validate=_validate_v2,
        bordered=True,
    )
    v2_form
    return (v2_form,)


@app.cell(hide_code=True)
def _(
    create_story_artifact,
    get_artifacts,
    get_disagreement_note,
    get_structure_note,
    mo,
    set_artifacts,
    text,
    utc_now_text,
    v2_form,
):
    _v2_card = mo.callout(mo.md(text["v2_waiting"]), kind="info")
    _v2_value = v2_form.value
    if _v2_value is not None:
        try:
            if get_structure_note() is None or get_disagreement_note() is None:
                raise ValueError(text["v2_waiting"])
            _artifact_items = get_artifacts()
            _first = next(_item for _item in _artifact_items if _item.version_label == "V1")
            _new_v2 = create_story_artifact(
                media_bytes=_v2_value["media"][0].contents,
                version_label="V2",
                parent_artifact_id=_first.artifact_id,
                local_registered_at_utc=utc_now_text(),
                event_index=1,
                elapsed_ms=0,
                creator_intention=_first.creator_intention,
                intended_audience=_first.intended_audience,
                sound_image_relation=_first.sound_image_relation,
                concept_tags=_first.concept_tags,
                cultural_aesthetic_context=_first.cultural_aesthetic_context,
                source_alias=str(_first.source_license_provenance["source_alias"]),
                source_permission_note=str(
                    _first.source_license_provenance["permission_or_provenance"]
                ),
                editor_name_version=_first.editor_name_version,
                ordering=" | ".join(str(_item) for _item in _first.edit_decisions["ordering"]),
                trims=" | ".join(
                    str(_item["decision"]) for _item in _first.edit_decisions["trims"]
                ),
                mix_levels=" | ".join(
                    str(_item["decision"]) for _item in _first.edit_decisions["mix_levels"]
                ),
                caption_or_transcript=_v2_value["caption"],
                visual_description=_v2_value["description"],
                overlays_human_reviewed=_v2_value["reviewed"],
                assistance_disclosure=str(_first.assistance_disclosure["disclosure"]),
                export_preset_version=_first.export_preset_version,
                change_rationale=_v2_value["change"],
            )
            _existing_v2 = next(
                (_item for _item in _artifact_items if _item.version_label == "V2"), None
            )
            if _existing_v2 is None:
                set_artifacts((_first, _new_v2))
            elif _existing_v2.artifact_id != _new_v2.artifact_id:
                raise ValueError("V2 is immutable; reset the session before replacing it")
        except Exception as _error:  # noqa: BLE001 — unsafe student input belongs in UI
            _v2_card = mo.callout(
                mo.md(f"**Registration rejected** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            _v2_card = mo.callout(mo.md(text["v2_complete"]), kind="success")
    _v2_card
    return


@app.cell(hide_code=True)
def _(mo, text):
    mo.md(
        f"""
        ## {text['section_finish']}

        ### {text['post_heading']}

        **Required / 필수.** {text['post_prompt']}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, text):
    def _validate_post(_value):
        if not _value or any(not str(_value[_key]).strip() for _key in _value):
            return text["post_submit"]
        return None

    post_task_form = mo.md(
        f"""
        **{text['post_evidence_label']}** {{evidence}}

        **{text['post_alternative_label']}** {{alternative}}

        **{text['post_limit_label']}** {{limit}}
        """
    ).batch(
        evidence=mo.ui.text_area(
            label=text["post_evidence_label"], rows=3, full_width=True
        ),
        alternative=mo.ui.text_area(
            label=text["post_alternative_label"], rows=3, full_width=True
        ),
        limit=mo.ui.text_area(
            label=text["post_limit_label"], rows=3, full_width=True
        ),
    ).form(
        submit_button_label=text["post_submit"],
        validate=_validate_post,
        bordered=True,
    )
    post_task_form
    return (post_task_form,)


@app.cell(hide_code=True)
def _(mo, text):
    fidelity_notes_form = mo.md(
        f"""
        **{text['timing_label']}** {{timing}}

        **{text['help_label']}** {{deviation}}
        """
    ).batch(
        timing=mo.ui.text_area(label=text["timing_label"], rows=2, full_width=True),
        deviation=mo.ui.text_area(label=text["help_label"], rows=2, full_width=True),
    ).form(
        submit_button_label="Record / 기록",
        bordered=True,
    )
    fidelity_notes_form
    return (fidelity_notes_form,)


@app.cell(hide_code=True)
def _(mo, saved_replay, text):
    _example = saved_replay["example"]
    mo.accordion(
        {
            text["replay_heading"]: mo.vstack(
                [
                    mo.md(text["replay_body"]),
                    mo.md(
                        "- "
                        + " → ".join(_example["story_beats"])
                        + f"\n- {_example['sound_role']}\n- {_example['accessibility_alternative']}"
                    ),
                ]
            )
        },
        multiple=False,
    )
    return


@app.cell(hide_code=True)
def _(
    build_private_story_bundle,
    fidelity_notes_form,
    get_artifacts,
    language,
    mo,
    post_task_form,
    pre_task_form,
    serialize_private_story_bundle,
    session_pseudonym,
    text,
):
    mo.md(f"### {text['export_heading']}\n\n{text['export_body']}")
    _export_ready = (
        tuple(_item.version_label for _item in get_artifacts()) == ("V1", "V2")
        and pre_task_form.value is not None
        and post_task_form.value is not None
    )
    if not _export_ready:
        _export_control = mo.callout(mo.md(text["export_waiting"]), kind="info")
    else:
        _notes = fidelity_notes_form.value or {"timing": "", "deviation": ""}
        try:
            _bundle = build_private_story_bundle(
                session_pseudonym=session_pseudonym,
                language=language,
                artifacts=get_artifacts(),
                pre_response=pre_task_form.value,
                post_response=post_task_form.value,
                stage_timing_notes=(_notes["timing"],),
                fidelity_deviations=(_notes["deviation"],),
            )
            _bundle_bytes = serialize_private_story_bundle(_bundle)
        except Exception as _error:  # noqa: BLE001 — export errors belong in UI
            _export_control = mo.callout(
                mo.md(f"**Export rejected** — `{type(_error).__name__}: {_error}`"),
                kind="danger",
            )
        else:
            _export_control = mo.download(
                data=_bundle_bytes,
                filename="digital-storytelling-private-bundle.json",
                label=text["export_download"],
            )
    _export_control
    return


if __name__ == "__main__":
    app.run()
