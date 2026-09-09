#!/usr/bin/env python
"""Render process/semantic readiness separately and check safe shutdown state."""
import argparse
import json
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-unreviewed',action='store_true')
    parser.add_argument('--status-only',action='store_true')
    parser.add_argument('--check-stop',action='store_true')
    args=parser.parse_args(argv)
    try: health=json.load(sys.stdin)
    except (ValueError,TypeError):
        print('상태 응답을 읽을 수 없습니다.');return 1
    if not isinstance(health,dict) or health.get('ok') is not True:
        print('서버 준비를 확인하지 못했습니다.');return 1
    counts=health.get('processing') or {}
    pending=sum(counts.get(state,0) for state in ('queued','splitting','embedding'))
    failed=counts.get('failed',0)
    context=health.get('class_context') or {}
    limits=health.get('limits') or {}
    reviewed=health.get('segmentation_reviewed') is True
    fake=str(health.get('cache_key','')).startswith('fake:')
    print(f"  명단: {health.get('roster_count','미확인')}명 · 주차: {context.get('week','미확인')} · 대상: {context.get('target_id') or '미지정'}")
    print(f"  입력 한도: {limits.get('max_raw_codepoints','미확인')} 코드 포인트 · 자동 의견: {limits.get('automatic_units','미확인')} · 대기 한도: {limits.get('max_pending','미확인')}")
    print(f"  처리 대기: {pending} · 실패: {failed} · 접수: {'열림' if context.get('accepting') else '닫힘/미확인'}")
    print(f"  모델 준비: {'완료' if health.get('model_warmed',health['ok']) else '미완료'} · 고급 보기: {'켜짐' if (health.get('viewer') or {}).get('enabled') else '꺼짐'}")
    print(f"  의미 분할 검토: {'승인' if reviewed else '미승인'}"+(' · 개발용 가짜 임베딩' if fake else ''))
    if args.check_stop:
        if context.get('accepting') is not False or pending or failed:
            print('접수를 닫고 처리 대기/실패 원문을 확인·복구한 뒤 종료하세요. 강제 종료는 stop --force입니다.')
            return 1
        return 0
    if args.status_only: return 0
    if not health.get('model_warmed',health['ok']): return 1
    if not reviewed or fake:
        if args.allow_unreviewed:
            print('  개발용 예외: 미승인 상태로 진행합니다. 수업 품질 승인을 뜻하지 않습니다.')
        else:
            print('의미 분할이 미승인인 상태로 수업 터널을 열지 않습니다. 개발 확인에만 ATLAS_ALLOW_UNREVIEWED_SEMANTICS=1을 사용하세요.')
            return 1
    return 0


if __name__=='__main__': raise SystemExit(main())
