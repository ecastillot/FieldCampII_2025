import os
import glob
from datetime import timedelta
from obspy.clients.filesystem.sds import Client
from obspy.core.util.misc import BAND_CODE

from obspy.core.util.misc import BAND_CODE
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError

import numpy as np

from obspy import Stream, read, UTCDateTime
from obspy.core.stream import _headonly_warning_msg
from obspy.core.util.misc import BAND_CODE
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError


class SmartSoloClient(Client):
    def __init__(self, root, fmt, **kwargs):
        self.root = root
        self.fmt = fmt
        super().__init__(root, **kwargs)

    def get_waveforms(self, station, component, starttime,
                      endtime, initial_station_name = "4530",merge=-1, **kwargs):
        """
        Read data from a local SeisComP Data Structure (SDS) directory tree.

        >>> from obspy import UTCDateTime
        >>> t = UTCDateTime("2015-10-12T12")
        >>> st = client.get_waveforms("IU", "ANMO", "*", "HH?", t, t+30)
        ... # doctest: +SKIP

        :type network: str
        :param network: Network code of requested data (e.g. "IU").
            Wildcards '*' and '?' are supported.
        :type station: str
        :param station: Station code of requested data (e.g. "ANMO").
            Wildcards '*' and '?' are supported.
        :type location: str
        :param location: Location code of requested data (e.g. "").
            Wildcards '*' and '?' are supported.
        :type channel: str
        :param channel: Channel code of requested data (e.g. "HHZ").
            Wildcards '*' and '?' are supported.
        :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param starttime: Start of requested time window.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`
        :param endtime: End of requested time window.
        :type merge: int or None
        :param merge: Specifies, which merge operation should be performed
            on the stream before returning the data. Default (``-1``) means
            only a conservative cleanup merge is performed to merge seamless
            traces (e.g. when reading across day boundaries). See
            :meth:`Stream.merge(...) <obspy.core.stream.Stream.merge>` for
            details. If set to ``None`` (or ``False``) no merge operation at
            all will be performed.
        :param kwargs: Additional kwargs that get passed on to
            :func:`~obspy.core.stream.read` internally, mostly for internal
            low-level purposes used by other methods.
        :rtype: :class:`~obspy.core.stream.Stream`
        """
        if starttime >= endtime:
            msg = ("'endtime' must be after 'starttime'.")
            raise ValueError(msg)

        st = Stream()
        full_paths = self._get_filenames(
            station=station, component=component,
            starttime=starttime, endtime=endtime,initial_station_name=initial_station_name)
        for full_path in full_paths:
            # print(f"Reading {full_path}")
            try:
                st += read(full_path, format=self.format, starttime=starttime,
                           endtime=endtime,  **kwargs)
                        #    endtime=endtime, sourcename=seed_pattern, **kwargs)
                # print(f"Reading {full_path}",seed_pattern,st)
            except ObsPyMSEEDFilesizeTooSmallError:
                # just ignore small MSEED files, in use cases working with
                # near-realtime data these are usually just being created right
                # at request time, e.g. when fetching current data right after
                # midnight
                continue
        
        # make sure we only have the desired data, just in case the file
        # contents do not match the expected SEED id
        stream = Stream()
        for sta in station.split(","):
            # print(sta,component,st)
            stream += st.select(station=sta, component=component)
            # print(sta,component,stream,"\n")
        st = stream

        # avoid trim/merge operations when we do a headonly read for
        # `_get_availability_percentage()`
        if kwargs.get("_no_trim_or_merge", False):
            return st

        st.trim(starttime, endtime)
        if merge is None or merge is False:
            pass
        else:
            st.merge(merge)
        return st

    def _get_filenames(self, station, component, starttime,
                       endtime,initial_station_name="4530"):

        t_buffer = self.fileborder_samples / BAND_CODE.get(component, 20.0)
        t_buffer = max(t_buffer, self.fileborder_seconds)

        t = starttime - t_buffer
        t_max = endtime + t_buffer

        time_steps = []
        while t <= t_max:
            time_steps.append(t)
            t += timedelta(seconds=1)
        full_paths = set()
        for t in time_steps:
            for sta in station.split(","):
                filename = self.fmt.format(
                    station=initial_station_name + sta,
                    year=t.year, month=t.month, day=t.day,
                    hour=t.hour, minute=t.minute, second=t.second,
                    milliseconds=0,
                    component=component  # assuming channel="EHZ", "BHZ", etc.
                )
                # print(filename)
                path = os.path.join(self.root, filename)
                full_paths.update(glob.glob(path))
        # print(full_paths)
        return full_paths

    def _get_filename(self, station, component, time):
        filename = self.fmt.format(
            station=station,
            year=time.year, month=time.month, day=time.day,
            hour=time.hour, minute=time.minute, second=time.second,
            milliseconds=int(time.microsecond / 1000),
            component=component
        )
        return os.path.join(self.root, filename)