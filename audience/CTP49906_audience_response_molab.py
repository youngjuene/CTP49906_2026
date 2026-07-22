# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo==0.23.14",
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
def _():
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from audience.response_core import (
        AUDIENCE_CONTENT,
        CLASSROOM_MINIMUM_NOTICE,
        build_reading,
        packet_for_display,
        parse_json_object,
    )
    from curriculum_common.audience_packets import (
        AUDIENCE_SHARING_PERMISSION,
        AudiencePacket,
        canonical_json_bytes,
    )

    return (
        AUDIENCE_CONTENT,
        AUDIENCE_SHARING_PERMISSION,
        AudiencePacket,
        CLASSROOM_MINIMUM_NOTICE,
        build_reading,
        canonical_json_bytes,
        packet_for_display,
        parse_json_object,
    )


@app.cell(hide_code=True)
def _(mo):
    language = mo.ui.dropdown(
        {"English": "en", "한국어": "ko"},
        value="en",
        label="Language / 언어",
    )
    language
    return (language,)


@app.cell
def _(AUDIENCE_CONTENT, language):
    content = AUDIENCE_CONTENT[language.value]
    return (content,)


@app.cell(hide_code=True)
def _(content, mo):
    mo.md(
        f"""
        # {content['title']}

        {content['purpose']}

        > **Data boundary:** {content['boundary']}
        """
    )
    return


@app.cell(hide_code=True)
def _(content, mo):
    def _validate_packet_upload(_value):
        if not _value:
            return "Choose one JSON packet."
        return None

    packet_upload = mo.ui.file(
        filetypes=[".json"],
        multiple=False,
        max_size=1_000_000,
        label=content["packet_upload"],
    ).form(
        submit_button_label=content["packet_submit"],
        validate=_validate_packet_upload,
        bordered=True,
    )
    mo.vstack([mo.md(f"## {content['packet_heading']}"), packet_upload])
    return (packet_upload,)


@app.cell
def _(AudiencePacket, packet_upload, parse_json_object):
    packet = None
    packet_error = None
    if packet_upload.value:
        try:
            _uploaded = packet_upload.value[0]
            packet = AudiencePacket.from_mapping(parse_json_object(_uploaded.contents))
        except (ValueError, UnicodeError) as _exc:
            packet_error = str(_exc)
    return packet, packet_error


@app.cell(hide_code=True)
def _(content, mo, packet, packet_error, packet_for_display):
    if packet_error:
        _packet_status = mo.md(f"**Packet rejected:** {packet_error}").callout(
            kind="danger"
        )
    elif packet is None:
        _packet_status = mo.md(
            "The response form stays locked until a packet passes validation."
        ).callout(kind="info")
    else:
        _display = packet_for_display(packet)
        _asset = _display["presentation_asset"]
        _packet_status = mo.vstack(
            [
                mo.md(f"**{content['packet_ready']}**").callout(kind="success"),
                mo.md(
                    f"""
                    - **Exchange:** `{_display['exchange_artifact_id']}`
                    - **Media:** `{_asset['media_type']}`, {_asset['duration_ms']} ms
                    - **Presentation reference:** `{_asset['asset_reference']}`
                    - **Caption reference:** `{_asset.get('caption_reference', 'not provided')}`
                    - **Language:** `{_asset.get('language', 'not specified')}`
                    - **Checksum:** `{_display['presentation_checksum']}`
                    """
                ),
            ]
        )
    _packet_status
    return


@app.cell(hide_code=True)
def _(content, mo, packet):
    def _validate_response(_value):
        if not _value:
            return "Complete the form."
        _required = (
            "audience_pseudonym",
            "session_pseudonym",
            "open_interpretation",
            "sound_image_relation",
            "accessibility_barriers",
            "confidence_or_ambiguity",
        )
        if any(not str(_value.get(_key, "")).strip() for _key in _required):
            return "Complete every required text field."
        if not _value.get("blindness_attestation"):
            return "Confirm blindness before committing a reading."
        return None

    response_form = (
        mo.md(
            """
            {audience_pseudonym}

            {session_pseudonym}

            {open_interpretation}

            {sound_image_relation}

            {shared_tags}

            {self_tags}

            {accessibility_barriers}

            {confidence_or_ambiguity}

            {blindness_attestation}
            """
        )
        .batch(
            audience_pseudonym=mo.ui.text(label=content["audience_pseudonym"]),
            session_pseudonym=mo.ui.text(label=content["session_pseudonym"]),
            open_interpretation=mo.ui.text_area(
                label=content["open_interpretation"], full_width=True
            ),
            sound_image_relation=mo.ui.text_area(
                label=content["sound_image_relation"], full_width=True
            ),
            shared_tags=mo.ui.text(label=content["shared_tags"], full_width=True),
            self_tags=mo.ui.text(label=content["self_tags"], full_width=True),
            accessibility_barriers=mo.ui.text_area(
                label=content["accessibility"], full_width=True
            ),
            confidence_or_ambiguity=mo.ui.text_area(
                label=content["confidence"], full_width=True
            ),
            blindness_attestation=mo.ui.checkbox(label=content["blindness"]),
        )
        .form(
            submit_button_label=content["response_submit"],
            submit_button_disabled=packet is None,
            validate=_validate_response,
            bordered=True,
        )
    )
    mo.vstack([mo.md(f"## {content['response_heading']}"), response_form])
    return (response_form,)


@app.cell
def _(
    AUDIENCE_SHARING_PERMISSION,
    build_reading,
    packet,
    response_form,
):
    committed_reading = None
    reading_error = None
    if packet is not None and response_form.value:
        _snapshot = response_form.value
        _split_tags = lambda _text: [
            _item.strip() for _item in str(_text).split(",") if _item.strip()
        ]
        try:
            committed_reading = build_reading(
                packet,
                {
                    "blindness_attestation": _snapshot["blindness_attestation"],
                    "permission_scope": AUDIENCE_SHARING_PERMISSION,
                    "open_interpretation": _snapshot["open_interpretation"],
                    "sound_image_relation": _snapshot["sound_image_relation"],
                    "shared_tags": _split_tags(_snapshot["shared_tags"]),
                    "self_described_tags": _split_tags(_snapshot["self_tags"]),
                    "accessibility_barriers": _snapshot["accessibility_barriers"],
                    "confidence_or_ambiguity": _snapshot["confidence_or_ambiguity"],
                    "withdrawn": False,
                },
                audience_pseudonym=_snapshot["audience_pseudonym"],
                respondent_session_pseudonym=_snapshot["session_pseudonym"],
                event_index=0,
                elapsed_ms=0,
            )
        except ValueError as _exc:
            reading_error = str(_exc)
    return committed_reading, reading_error


@app.cell(hide_code=True)
def _(canonical_json_bytes, committed_reading, content, mo, reading_error):
    if reading_error:
        _reading_status = mo.md(f"**Reading rejected:** {reading_error}").callout(
            kind="danger"
        )
    elif committed_reading is None:
        _reading_status = mo.md(
            "No response has been committed. Editing the draft does not create a record."
        ).callout(kind="neutral")
    else:
        _reading_bytes = canonical_json_bytes(committed_reading.to_dict()) + b"\n"
        _download = mo.download(
            data=_reading_bytes,
            filename=f"{committed_reading.response_id}.json",
            mimetype="application/json",
            label=content["download"],
        )
        _reading_status = mo.vstack(
            [
                mo.md(f"**{content['complete']}**").callout(kind="success"),
                _download,
            ]
        )
    _reading_status
    return


@app.cell(hide_code=True)
def _(CLASSROOM_MINIMUM_NOTICE, mo):
    mo.md(
        f"""
        ## Before reveal

        Keep each response independent. Do not show a prior audience response,
        creator explanation, condition, or machine label to the next respondent.
        The creator-facing notebook unlocks comparison only after two distinct,
        schema-valid blinded response files are imported.

        **Interpretation limit:** {CLASSROOM_MINIMUM_NOTICE}

        Captions, transcripts, and descriptions remain human-correctable access
        supports. This surface does not send them to a model or silently turn them
        into model input.
        """
    )
    return


if __name__ == "__main__":
    app.run()
