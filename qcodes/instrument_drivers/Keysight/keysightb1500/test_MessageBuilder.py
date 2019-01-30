from qcodes.instrument_drivers.Keysight.keysightb1500 import \
    MessageBuilder

import qcodes.instrument_drivers.Keysight.keysightb1500.constants as c

import pytest


@pytest.fixture
def b1500() -> MessageBuilder:
    yield MessageBuilder()



def test_as_csv():
    from qcodes.instrument_drivers.Keysight.keysightb1500.message_builder \
        import as_csv

    assert '1' == as_csv([1])

    assert '1,2,3' == as_csv([1, 2, 3])

    assert '1,2' == as_csv([c.ChNr.SLOT_01_CH1, c.ChNr.SLOT_02_CH1])


class TestMessageBuilder:
    def skip(self):
        pytest.skip("not implemented yet")

    def test_cmd(self, b1500):
        c = b1500.aad(1, 0).ach(1, 5).ab()
        assert 'AAD 1,0;ACH 1,5;AB' == c.message

    def test_raise_error_on_appending_command_after_final_command(self, b1500):
        with pytest.raises(ValueError):
            c = b1500.aad(1, 0).ab().ach(1, 5)

    def test_warning_on_too_long_message(self, b1500):
        length = 0
        while length < 250:
            b1500.os()
            length += 3  # "OS;" <- three characters per iteration

        with pytest.warns(UserWarning):
            assert len(b1500.message) > 250

    def test_aad(self, b1500):
        assert 'AAD 1,0' == \
               b1500.aad(c.ChNr.SLOT_01_CH1,
                         c.AAD.Type.HIGH_SPEED).message

        assert 'AAD 1,1' == \
               b1500.aad(c.ChNr.SLOT_01_CH1,
                         adc_type=c.AAD.Type.HIGH_RESOLUTION).message

    def test_ab(self, b1500):
        assert 'AB' == b1500.ab().message

    def test_ach(self, b1500):
        assert 'ACH 1,5' == b1500.ach(c.ChNr.SLOT_01_CH1,
                                      c.ChNr.SLOT_05_CH1).message

        assert 'ACH 1' == b1500.ach(c.ChNr.SLOT_01_CH1).message

        assert 'ACH' == b1500.ach().message

    def test_act(self, b1500):
        assert 'ACT 0,1' == b1500.act(c.ACT.Mode.AUTO, 1).message
        assert 'ACT 2' == b1500.act(c.ACT.Mode.PLC).message

    def test_acv(self, b1500):
        assert 'ACV 7,0.01' == \
               b1500.acv(c.ChNr.SLOT_07_CH1, 0.01).message

    def test_adj(self, b1500):
        assert 'ADJ 9,1' == b1500.adj(c.ChNr.SLOT_09_CH1,
                                      c.ADJ.Mode.MANUAL).message

    def test_adj_query(self, b1500):
        assert 'ADJ? 1' == b1500.adj_query(c.ChNr.SLOT_01_CH1).message

        assert 'ADJ? 1,1' == b1500.adj_query(c.ChNr.SLOT_01_CH1,
                                             c.ADJQuery.Mode.MEASURE).message

    def test_ait(self, b1500):
        assert 'AIT 2,3,0.001' == b1500.ait(c.AIT.Type.HIGH_SPEED_PULSED,
                                            c.AIT.Mode.MEAS_TIME_MODE,
                                            0.001).message

        assert 'AIT 2,3' == b1500.ait(c.AIT.Type.HIGH_SPEED_PULSED,
                                      c.AIT.Mode.MEAS_TIME_MODE).message

    def test_aitm(self, b1500):
        assert 'AITM 0' == b1500.aitm(c.APIVersion.B1500).message

    def test_aitm_query(self, b1500: MessageBuilder):
        assert 'AITM?' == b1500.aitm_query().message

    def test_als(self, b1500):
        with pytest.raises(NotImplementedError):
            b1500.als(c.ChNr.SLOT_01_CH1, 1, b'a')

    def test_als_query(self, b1500):
        assert 'ALS? 1' == b1500.als_query(c.ChNr.SLOT_01_CH1).message

    def test_alw(self, b1500):
        with pytest.raises(NotImplementedError):
            b1500.alw(c.ChNr.SLOT_01_CH1, 1, b'a')

    def test_alw_query(self, b1500):
        assert 'ALW? 1' == b1500.alw_query(c.ChNr.SLOT_01_CH1).message

    def test_av(self, b1500):
        assert 'AV 10' == b1500.av(10).message

        assert 'AV -50' == b1500.av(-50).message

        assert 'AV 100,1' == b1500.av(100, c.AV.Mode.MANUAL).message

    def test_az(self, b1500):
        assert 'AZ 0' == b1500.az(False).message

    def test_bc(self, b1500):
        assert 'BC' == b1500.bc().message

    def test_bdm(self, b1500: MessageBuilder):
        assert 'BDM 0,1' == b1500.bdm(c.BDM.Interval.SHORT,
                                      c.BDM.Mode.CURRENT).message

    def test_bdt(self, b1500: MessageBuilder):
        assert 'BDT 0.1,0.001' == b1500.bdt(hold=0.1, delay=1e-3).message

    def test_bdv(self, b1500):
        assert 'BDV 1,0,0,100,0.01' == b1500.bdv(chnum=c.ChNr.SLOT_01_CH1,
                                                 v_range=c.VOutputRange.AUTO,
                                                 start=0, stop=100,
                                                 i_comp=0.01).message

    def test_bgi(self, b1500):
        assert 'BGI 1,0,1e-08,14,1e-06' == \
               b1500.bgi(chnum=c.ChNr.SLOT_01_CH1,
                         searchmode=c.BinarySearchMode.LIMIT,
                         stop_condition=1e-8,
                         i_range=14,
                         target=1e-6).message

    def test_bgv(self, b1500):
        assert 'BGV 1,0,0.1,12,5' == b1500.bgv(1, 0, 0.1, 12, 5).message

    def test_bsi(self, b1500):
        assert 'BSI 1,0,1e-12,1e-06,10' == b1500.bsi(1, 0, 1e-12, 1e-6,
                                                     10).message

    def test_bsm(self, b1500):
        assert 'BSM 1,2,3' == b1500.bsm(1, 2, 3).message

    def test_bssi(self, b1500):
        assert 'BSSI 1,0,1e-06,10' == b1500.bssi(1, 0, 1e-6, 10).message

    def test_bssv(self, b1500):
        assert 'BSSV 1,0,5,1e-06' == b1500.bssv(1, 0, 5, 1e-6).message

    def test_bst(self, b1500):
        assert 'BST 5,0.1' == b1500.bst(5, 0.1).message

    def test_bsv(self, b1500):
        assert 'BSV 1,0,0,20,1e-06' == b1500.bsv(1, 0, 0, 20, 1e-6).message

    def test_bsvm(self, b1500):
        assert 'BSVM 1' == b1500.bsvm(1).message

    def test_ca(self, b1500):
        assert 'CA' == b1500.ca().message

    def test_cal_query(self, b1500):
        assert '*CAL?' == b1500.cal_query().message

    def test_cl(self, b1500):
        assert 'CL' == b1500.cl().message
        assert 'CL 1,2,3,5' == b1500.cl([1, 2, 3, 5]).message

    def test_clcorr(self, b1500):
        assert 'CLCORR 9,1' == b1500.clcorr(9, 1).message

    def test_cm(self, b1500):
        assert 'CM 0' == b1500.cm(0).message

    def test_cmm(self, b1500):
        assert 'CMM 1,2' == b1500.cmm(1, 2).message

    def test_cn(self, b1500):
        assert 'CN' == b1500.cn().message
        assert 'CN 1,2,3,5' == b1500.cn([1, 2, 3, 5]).message

    def test_cnx(self, b1500):
        assert 'CNX' == b1500.cnx().message
        assert 'CNX 1,2,3,5' == b1500.cnx([1, 2, 3, 5]).message

    def test_corr_query(self, b1500):
        assert 'CORR? 9,3' == b1500.corr_query(9, 3).message

    def test_corrdt(self, b1500):
        assert 'CORRDT 9,3000000,0,0,0,0,0,0' == b1500.corrdt(9, 3000000, 0,
                                                              0, 0, 0, 0,
                                                              0).message

    def test_corrdt_query(self, b1500):
        assert 'CORRDT? 9,1' == b1500.corrdt_query(9, 1).message

    def test_corrl(self, b1500):
        assert 'CORRL 9,3000000' == b1500.corrl(9, 3000000).message

    def test_corrl_query(self, b1500):
        assert 'CORRL? 9' == b1500.corrl_query(9).message
        assert 'CORRL? 9' == b1500.corrl_query(9).message

    def test_corrser_query(self, b1500: MessageBuilder):
        assert 'CORRSER? 101,1,1e-07,1e-08,10' == \
               b1500.corrser_query(101,
                                   True,
                                   1E-7,
                                   1E-8,
                                   10).message

    def test_corrst(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_corrst_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_dcorr(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_dcorr_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_dcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_di(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_diag_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_do(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_dsmplarm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_dsmplflush(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_dsmplsetup(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_dv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_dz(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_emg_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_end(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ercmaa(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ercmaa_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ercmagrd(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ercmagrd_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ercmaio(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ercmaio_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ercmapfgd(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpa(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpa_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpe(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpe_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpl(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpl_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpp_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpqg(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpqg_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpr(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhpr_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhps(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhps_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhvca(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhvca_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhvctst_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhvp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhvp_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhvpv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhvs(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erhvs_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ermod(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ermod_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfda(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfda_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfdp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfdp_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfds(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfds_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfga(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfga_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfgp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfgp_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfgr(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfgr_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfqg(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfqg_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpftemp_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfuhca(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfuhca_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfuhccal_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfuhcmax_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erpfuhctst(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_err_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_errx_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ers_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erssp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_erssp_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_eruhva(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_eruhva_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_fc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_fl(self, b1500):
        assert 'FL 1' == b1500.fl(True).message
        assert 'FL 0,1,3,5' == b1500.fl(False, [1, 3, 5]).message
        assert 'FL 0,1,2,3,4,5,6,7,8,9,10' == \
               b1500.fl(False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).message

        with pytest.raises(ValueError):
            x = b1500.fl(False, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]).message

    def test_fmt(self, b1500):
        assert 'FMT 1' == \
               b1500.fmt(c.FMT.Format.ASCII_12_DIGITS_WITH_HEADER_CRLF_EOI
                         ).message

        assert 'FMT 2,1' == \
               b1500.fmt(c.FMT.Format.ASCII_12_DIGITS_NO_HEADER_CRLF_EOI,
                         c.FMT.Mode.PRIMARY_SOURCE_OUTPUT_DATA).message

    def test_hvsmuop(self, b1500):
        self.skip()

    def test_hvsmuop_query(self, b1500):
        assert 'HVSMUOP?' == \
               b1500.hvsmuop_query().message

    def test_idn_query(self, b1500):
        assert 'IN' == b1500.in_().message
        assert 'IN 1,2,3,5,6' == b1500.in_([1, 2, 3, 5, 6]).message

    def test_imp(self, b1500):
        assert 'IMP 10' == b1500.imp(c.IMP.MeasurementMode.Z_THETA_RAD).message

    def test_in_(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_intlkvth(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_intlkvth_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lgi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lgv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lim(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lim_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lmn(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lop_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lrn_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lsi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lsm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lssi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lssv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lst_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lstm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lsv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_lsvm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mcc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mcpnt(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mcpnx(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mcpt(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mcpws(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mcpwnx(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ml(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mm(self, b1500):
        assert 'MM 1,1' == \
               b1500.mm(mode=c.MM.Mode.SPOT,
                        channels=[c.ChNr.SLOT_01_CH1]).message

        assert 'MM 2,1,3' == b1500.mm(mode=c.MM.Mode.STAIRCASE_SWEEP,
                                      channels=[c.ChNr.SLOT_01_CH1,
                                                c.ChNr.SLOT_03_CH1]).message

    def test_msc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_msp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mt(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mtdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_mv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_nub_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_odsw(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_odsw_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_opc_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_os(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_osx(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pa(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pad(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pax(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pch(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pch_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pt(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ptdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pwdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pwi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_pwv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_qsc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_qsl(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_qsm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_qso(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_qsr(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_qst(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_qsv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_qsz(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_rc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_rcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ri(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_rm(self, b1500):
        assert 'RM 1,2' == b1500.rm(1, c.RM.Mode.AUTO_UP).message
        assert 'RM 2,3,60' == b1500.rm(2, c.RM.Mode.AUTO_UP_DOWN, 60).message

        with pytest.raises(ValueError):
            b1500.rm(c.ChNr.SLOT_01_CH1, c.RM.Mode.DEFAULT, 22)

    def test_rst(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ru(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_rv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_rz(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sal(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sap(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sar(self, b1500):
        assert 'SAR 1,0' == b1500.sar(1,
                                      enable_picoamp_autoranging=True).message

    def test_scr(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ser(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ser_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sim(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sim_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sopc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sopc_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sovc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sovc_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spm_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spper(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spper_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sprm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sprm_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spst_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spt(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spt_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spupd(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_spv_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sre(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_sre_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_srp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ssl(self, b1500):
        assert 'SSL 9,0' == b1500.ssl(9, enable_indicator_led=False).message

    def test_ssp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ssr(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_st(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_stb_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_stgp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_stgp_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tacv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tdi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tdv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tgmo(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tgp(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tgpc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tgsi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tgso(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tgxo(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ti(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tiv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tmacv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tmdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tsc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tsq(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tsr(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tst(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ttc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tti(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ttiv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ttv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_tv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_unt_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_var(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_var_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wacv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wat(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wfc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wm(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wmacv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wmdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wmfc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wncc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wnu_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wnx(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_ws(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wsi(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wsv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wt(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wtacv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wtdcv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wtfc(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wv(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_wz_query(self, b1500):
        # assert '' == b1500.().message
        self.skip()

    def test_xe(self, b1500):
        # assert '' == b1500.().message
        self.skip()
