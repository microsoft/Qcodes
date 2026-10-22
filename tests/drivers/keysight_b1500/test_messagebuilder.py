import re
from typing import TYPE_CHECKING, NoReturn

import pytest

import qcodes.instrument_drivers.Keysight.keysightb1500.constants as c
from qcodes.instrument_drivers.Keysight.keysightb1500.message_builder import (
    MessageBuilder,
    as_csv,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def mb() -> "Generator[MessageBuilder, None, None]":
    yield MessageBuilder()


def test_as_csv() -> None:
    assert "1" == as_csv([1])

    assert "1,2,3" == as_csv([1, 2, 3])

    assert "1,2" == as_csv([c.ChNr.SLOT_01_CH1, c.ChNr.SLOT_02_CH1])


def _skip() -> NoReturn:
    pytest.skip("not implemented yet")


def test_cmd(mb: MessageBuilder) -> None:
    cmd = mb.aad(1, 0).ach(1, 5).ab()
    assert "AAD 1,0;ACH 1,5;AB" == cmd.message


def test_raise_error_on_appending_command_after_final_command(
    mb: MessageBuilder,
) -> None:
    with pytest.raises(ValueError):
        mb.aad(1, 0).ab().ach(1, 5)


def test_exception_on_too_long_message(mb: MessageBuilder) -> None:
    length = 0
    while length < 250:
        mb.os()
        length += 3  # "OS;" <- three characters per iteration

    match = re.escape(
        f"Command is too long ({len(str(mb._msg))}>256-termchars) "
        f"and will overflow input buffer of instrument. "
        f"(Consider using the ST/END/DO/RU commands for very long "
        f"programs.)"
    )
    with pytest.raises(Exception, match=match):
        _ = mb.message


def test_clear_message_queue(mb: MessageBuilder) -> None:
    mb.aad(1, 0)
    assert mb.message == "AAD 1,0"

    mb.clear_message_queue()

    assert mb.message == ""


def test_aad(mb: MessageBuilder) -> None:
    assert "AAD 1,0" == mb.aad(c.ChNr.SLOT_01_CH1, c.AAD.Type.HIGH_SPEED).message

    mb.clear_message_queue()

    assert (
        "AAD 1,1"
        == mb.aad(c.ChNr.SLOT_01_CH1, adc_type=c.AAD.Type.HIGH_RESOLUTION).message
    )


def test_ab(mb: MessageBuilder) -> None:
    assert "AB" == mb.ab().message


def test_ach(mb: MessageBuilder) -> None:
    assert "ACH 1,5" == mb.ach(c.ChNr.SLOT_01_CH1, c.ChNr.SLOT_05_CH1).message

    mb.clear_message_queue()

    assert "ACH 1" == mb.ach(c.ChNr.SLOT_01_CH1).message

    mb.clear_message_queue()

    assert "ACH" == mb.ach().message


def test_act(mb: MessageBuilder) -> None:
    assert "ACT 0,1" == mb.act(c.ACT.Mode.AUTO, 1).message

    mb.clear_message_queue()

    assert "ACT 2" == mb.act(c.ACT.Mode.PLC).message


def test_acv(mb: MessageBuilder) -> None:
    assert "ACV 7,0.01" == mb.acv(c.ChNr.SLOT_07_CH1, 0.01).message


def test_adj(mb: MessageBuilder) -> None:
    assert "ADJ 9,1" == mb.adj(c.ChNr.SLOT_09_CH1, c.ADJ.Mode.MANUAL).message


def test_adj_query(mb: MessageBuilder) -> None:
    assert "ADJ? 1" == mb.adj_query(c.ChNr.SLOT_01_CH1).message

    mb.clear_message_queue()

    assert (
        "ADJ? 1,1" == mb.adj_query(c.ChNr.SLOT_01_CH1, c.ADJQuery.Mode.MEASURE).message
    )


def test_ait(mb: MessageBuilder) -> None:
    assert (
        "AIT 2,3,0.001"
        == mb.ait(
            c.AIT.Type.HIGH_SPEED_PULSED, c.AIT.Mode.MEAS_TIME_MODE, 0.001
        ).message
    )
    mb.clear_message_queue()
    assert (
        "AIT 2,3"
        == mb.ait(c.AIT.Type.HIGH_SPEED_PULSED, c.AIT.Mode.MEAS_TIME_MODE).message
    )


def test_aitm(mb: MessageBuilder) -> None:
    assert "AITM 0" == mb.aitm(c.APIVersion.B1500).message


def test_aitm_query(mb: MessageBuilder) -> None:
    assert "AITM?" == mb.aitm_query().message


def test_als(mb: MessageBuilder) -> None:
    with pytest.raises(NotImplementedError):
        mb.als(c.ChNr.SLOT_01_CH1, 1, b"a")


def test_als_query(mb: MessageBuilder) -> None:
    assert "ALS? 1" == mb.als_query(c.ChNr.SLOT_01_CH1).message


def test_alw(mb: MessageBuilder) -> None:
    with pytest.raises(NotImplementedError):
        mb.alw(c.ChNr.SLOT_01_CH1, 1, b"a")


def test_alw_query(mb: MessageBuilder) -> None:
    assert "ALW? 1" == mb.alw_query(c.ChNr.SLOT_01_CH1).message


def test_av(mb: MessageBuilder) -> None:
    assert "AV 10" == mb.av(10).message
    mb.clear_message_queue()
    assert "AV -50" == mb.av(-50).message
    mb.clear_message_queue()
    assert "AV 100,1" == mb.av(100, c.AV.Mode.MANUAL).message


def test_az(mb: MessageBuilder) -> None:
    assert "AZ 0" == mb.az(False).message
    mb.clear_message_queue()
    assert "AZ 1" == mb.az(True).message


def test_bc(mb: MessageBuilder) -> None:
    assert "BC" == mb.bc().message


def test_bdm(mb: MessageBuilder) -> None:
    assert "BDM 0,1" == mb.bdm(c.BDM.Interval.SHORT, c.BDM.Mode.CURRENT).message


def test_bdt(mb: MessageBuilder) -> None:
    assert "BDT 0.1,0.001" == mb.bdt(hold=0.1, delay=1e-3).message


def test_bdv(mb: MessageBuilder) -> None:
    assert (
        "BDV 1,0,0,100,0.01"
        == mb.bdv(
            chnum=c.ChNr.SLOT_01_CH1,
            v_range=c.VOutputRange.AUTO,
            start=0,
            stop=100,
            i_comp=0.01,
        ).message
    )


def test_bgi(mb: MessageBuilder) -> None:
    assert (
        "BGI 1,0,1e-08,14,1e-06"
        == mb.bgi(
            chnum=c.ChNr.SLOT_01_CH1,
            searchmode=c.BinarySearchMode.LIMIT,
            stop_condition=1e-8,
            i_range=14,
            target=1e-6,
        ).message
    )


def test_bgv(mb: MessageBuilder) -> None:
    assert "BGV 1,0,0.1,12,5" == mb.bgv(1, 0, 0.1, 12, 5).message


def test_bsi(mb: MessageBuilder) -> None:
    assert "BSI 1,0,1e-12,1e-06,10" == mb.bsi(1, 0, 1e-12, 1e-6, 10).message


def test_bsm(mb: MessageBuilder) -> None:
    assert "BSM 1,2,3" == mb.bsm(1, 2, 3).message


def test_bssi(mb: MessageBuilder) -> None:
    assert "BSSI 1,0,1e-06,10" == mb.bssi(1, 0, 1e-6, 10).message


def test_bssv(mb: MessageBuilder) -> None:
    assert "BSSV 1,0,5,1e-06" == mb.bssv(1, 0, 5, 1e-6).message


def test_bst(mb: MessageBuilder) -> None:
    assert "BST 5,0.1" == mb.bst(5, 0.1).message


def test_bsv(mb: MessageBuilder) -> None:
    assert "BSV 1,0,0,20,1e-06" == mb.bsv(1, 0, 0, 20, 1e-6).message


def test_bsvm(mb: MessageBuilder) -> None:
    assert "BSVM 1" == mb.bsvm(1).message


def test_ca(mb: MessageBuilder) -> None:
    assert "CA" == mb.ca().message


def test_cal_query(mb: MessageBuilder) -> None:
    assert "*CAL?" == mb.cal_query().message


def test_cl(mb: MessageBuilder) -> None:
    assert "CL" == mb.cl().message
    mb.clear_message_queue()
    assert "CL 1,2,3,5" == mb.cl([1, 2, 3, 5]).message


def test_clcorr(mb: MessageBuilder) -> None:
    assert "CLCORR 9,1" == mb.clcorr(9, 1).message


def test_cm(mb: MessageBuilder) -> None:
    assert "CM 0" == mb.cm(False).message


def test_cmm(mb: MessageBuilder) -> None:
    assert "CMM 1,2" == mb.cmm(1, 2).message


def test_cn(mb: MessageBuilder) -> None:
    assert "CN" == mb.cn().message
    mb.clear_message_queue()
    assert "CN 1,2,3,5" == mb.cn([1, 2, 3, 5]).message


def test_cnx(mb: MessageBuilder) -> None:
    assert "CNX" == mb.cnx().message
    mb.clear_message_queue()
    assert "CNX 1,2,3,5" == mb.cnx([1, 2, 3, 5]).message


def test_corr_query(mb: MessageBuilder) -> None:
    assert "CORR? 9,3" == mb.corr_query(9, 3).message


def test_corrdt(mb: MessageBuilder) -> None:
    assert (
        "CORRDT 9,3000000,0,0,0,0,0,0"
        == mb.corrdt(9, 3000000, 0, 0, 0, 0, 0, 0).message
    )


def test_corrdt_query(mb: MessageBuilder) -> None:
    assert "CORRDT? 9,1" == mb.corrdt_query(9, 1).message


def test_corrl(mb: MessageBuilder) -> None:
    assert "CORRL 9,3000000" == mb.corrl(9, 3000000).message


def test_corrl_query(mb: MessageBuilder) -> None:
    assert "CORRL? 9" == mb.corrl_query(9).message
    mb.clear_message_queue()
    assert "CORRL? 9,2" == mb.corrl_query(9, 2).message


def test_corrser_query(mb: MessageBuilder) -> None:
    assert (
        "CORRSER? 101,1,1e-07,1e-08,10"
        == mb.corrser_query(101, True, 1e-7, 1e-8, 10).message
    )


def test_corrst(mb: MessageBuilder) -> None:
    assert "CORRST 3,1,1" == mb.corrst(3, 1, True).message


def test_corrst_query(mb: MessageBuilder) -> None:
    assert "CORRST? 1,2" == mb.corrst_query(1, 2).message


def test_dcorr(mb: MessageBuilder) -> None:
    assert "DCORR 3,1,100,0.1,0.2" == mb.dcorr(3, 1, 100, 0.1, 0.2).message


def test_dcorr_query(mb: MessageBuilder) -> None:
    assert "DCORR? 3,1" == mb.dcorr_query(3, 1).message


def test_dcv(mb: MessageBuilder) -> None:
    assert "DCV 3,2.5" == mb.dcv(3, 2.5).message


def test_di(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_diag_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_do(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_dsmplarm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_dsmplflush(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_dsmplsetup(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_dv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_dz(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_emg_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_end(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmaa(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmaa_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmagrd(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmagrd_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmaio(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmaio_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ercmapfgd(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpa(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpa_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpe(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpe_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpl(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpl_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpp_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpqg(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpqg_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpr(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhpr_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhps(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhps_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvca(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvca_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvctst_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvp_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvpv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvs(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erhvs_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ermod(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ermod_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfda(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfda_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfdp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfdp_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfds(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfds_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfga(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfga_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfgp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfgp_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfgr(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfgr_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfqg(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfqg_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpftemp_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhca(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhca_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhccal_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhcmax_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erpfuhctst(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_err_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_errx_query(mb: MessageBuilder) -> None:
    assert "ERRX?" == mb.errx_query().message
    mb.clear_message_queue()
    assert "ERRX? 0" == mb.errx_query(mode=c.ERRX.Mode.CODE_AND_MESSAGE).message


def test_ers_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erssp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_erssp_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_eruhva(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_eruhva_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_fc(mb: MessageBuilder) -> None:
    assert "FC 3,1000" == mb.fc(3, 1000).message


def test_fl(mb: MessageBuilder) -> None:
    assert "FL 1" == mb.fl(True).message
    mb.clear_message_queue()
    assert "FL 0,1,3,5" == mb.fl(False, [1, 3, 5]).message
    mb.clear_message_queue()
    assert (
        "FL 0,1,2,3,4,5,6,7,8,9,10"
        == mb.fl(False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).message
    )
    mb.clear_message_queue()
    with pytest.raises(ValueError):
        _ = mb.fl(False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).message


def test_fmt(mb: MessageBuilder) -> None:
    assert "FMT 1" == mb.fmt(c.FMT.Format.ASCII_12_DIGITS_WITH_HEADER_CRLF_EOI).message
    mb.clear_message_queue()
    assert (
        "FMT 2,1"
        == mb.fmt(
            c.FMT.Format.ASCII_12_DIGITS_NO_HEADER_CRLF_EOI,
            c.FMT.Mode.PRIMARY_SOURCE_OUTPUT_DATA,
        ).message
    )


def test_hvsmuop(mb: MessageBuilder) -> None:
    _skip()


def test_hvsmuop_query(mb: MessageBuilder) -> None:
    assert "HVSMUOP?" == mb.hvsmuop_query().message


def test_idn_query(mb: MessageBuilder) -> None:
    assert "IN" == mb.in_().message
    mb.clear_message_queue()
    assert "IN 1,2,3,5,6" == mb.in_([1, 2, 3, 5, 6]).message


def test_imp(mb: MessageBuilder) -> None:
    assert "IMP 10" == mb.imp(c.IMP.MeasurementMode.Z_THETA_RAD).message


def test_in_(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_intlkvth(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_intlkvth_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lgi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lgv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lim(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lim_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lmn(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lop_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lrn_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lsi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lsm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lssi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lssv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lst_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lstm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lsv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_lsvm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpnt(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpnx(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpt(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpws(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mcpwnx(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ml(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mm(mb: MessageBuilder) -> None:
    assert "MM 1,1" == mb.mm(mode=c.MM.Mode.SPOT, channels=[c.ChNr.SLOT_01_CH1]).message
    mb.clear_message_queue()
    assert (
        "MM 2,1,3"
        == mb.mm(
            mode=c.MM.Mode.STAIRCASE_SWEEP,
            channels=[c.ChNr.SLOT_01_CH1, c.ChNr.SLOT_03_CH1],
        ).message
    )


def test_msc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_msp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mt(mb: MessageBuilder) -> None:
    assert "MT 0.0,0.42,32" == mb.mt(0.0, 0.42, 32).message
    mb.clear_message_queue()
    assert "MT 0.0,0.42,32,0.12" == mb.mt(0.0, 0.42, 32, 0.12).message


def test_mtdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_mv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_nub_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_odsw(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_odsw_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_opc_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_os(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_osx(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pa(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pad(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pax(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pch(mb: MessageBuilder) -> None:
    assert "PCH 1,2" == mb.pch(1, 2).message


def test_pch_query(mb: MessageBuilder) -> None:
    assert "PCH?" == mb.pch_query().message


def test_pch_query_with_master(mb: MessageBuilder) -> None:
    assert "PCH? 1" == mb.pch_query(1).message


def test_pdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pt(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ptdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pwdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pwi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_pwv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsl(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_qso(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsr(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_qst(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_qsz(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_rc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_rcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ri(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_rm(mb: MessageBuilder) -> None:
    assert "RM 1,2" == mb.rm(1, c.RM.Mode.AUTO_UP).message
    mb.clear_message_queue()
    assert "RM 2,3,60" == mb.rm(2, c.RM.Mode.AUTO_UP_DOWN, 60).message
    mb.clear_message_queue()
    with pytest.raises(ValueError):
        mb.rm(c.ChNr.SLOT_01_CH1, c.RM.Mode.DEFAULT, 22)


def test_rst(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ru(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_rv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_rz(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sal(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sap(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sar(mb: MessageBuilder) -> None:
    assert "SAR 1,0" == mb.sar(1, enable_picoamp_autoranging=True).message


def test_scr(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ser(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ser_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sim(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sim_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sopc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sopc_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sovc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sovc_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spm_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spper(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spper_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sprm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sprm_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spst_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spt(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spt_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spupd(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_spv_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sre(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_sre_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_srp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ssl(mb: MessageBuilder) -> None:
    assert "SSL 9,0" == mb.ssl(9, enable_indicator_led=False).message


def test_ssp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ssr(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_st(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_stb_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_stgp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_stgp_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tacv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tc(mb: MessageBuilder) -> None:
    assert "TC 3,0" == mb.tc(3, 0).message


def test_tdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tdi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tdv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgmo(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgp(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgpc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgsi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgso(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tgxo(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ti(mb: MessageBuilder) -> None:
    assert "TI 1" == mb.ti(chnum=1).message
    mb.clear_message_queue()
    assert "TI 1,-14" == mb.ti(chnum=1, i_range=c.IMeasRange.FIX_1uA).message


def test_tiv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tmacv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tmdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tsc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tsq(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tsr(mb: MessageBuilder) -> None:
    assert "TSR" == mb.tsr().message
    mb.clear_message_queue()
    assert "TSR 1" == mb.tsr(c.ChNr.SLOT_01_CH1).message


def test_tst(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ttc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tti(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ttiv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ttv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_tv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_unt_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_var(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_var_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wacv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wat(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wfc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wm(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wmacv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wmdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wmfc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wncc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wnu_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wnx(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_ws(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wsi(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wsv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wt(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wtacv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wtdcv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wtfc(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wv(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_wz_query(mb: MessageBuilder) -> None:
    # assert '' == mb.().message
    _skip()


def test_xe(mb: MessageBuilder) -> None:
    assert "XE" == mb.xe().message


def test_nplc_setting_for_high_speed_vs_high_resolution_mode(
    mb: MessageBuilder,
) -> None:
    msg = (
        mb.ait(adc_type=c.AIT.Type.HIGH_SPEED, mode=c.AIT.Mode.NPLC, coeff=3)
        .ait(adc_type=c.AIT.Type.HIGH_RESOLUTION, mode=c.AIT.Mode.NPLC, coeff=8)
        .message
    )
    assert msg == "AIT 0,2,3;AIT 1,2,8"


def test_set_resolution_mode_for_each_smu(mb: MessageBuilder) -> None:
    msg = (
        mb.aad(chnum=1, adc_type=c.AAD.Type.HIGH_SPEED)
        .aad(chnum=2, adc_type=c.AAD.Type.HIGH_RESOLUTION)
        .message
    )
    assert msg == "AAD 1,0;AAD 2,1"
