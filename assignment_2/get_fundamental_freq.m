% Helper function for getting the fundamental frequency of note. This
% function is used to help filter out the overtones in our music data in
% the frequency domain.
% Takes in a note, array of fundamental frequencies in the first harmonic
% to compare note too, and a specified error to compare with.
function music_note = get_fundamental_freq(note, fund_freqs, err)
   music_note = 0;
   for i = 1:length(fund_freqs)
       fund_freq_lower = fund_freqs(i) - err;
       fund_freq_higher = fund_freqs(i) + err;
       % First harmonics
       if ((fund_freq_lower <= note) && (note <= fund_freq_higher))
           music_note = fund_freqs(i);

       % Second Harmonic
       elseif ((fund_freq_lower <= note/2) && (note/2 <= fund_freq_higher))
           music_note = fund_freqs(i);
       
       % Third Harmonic
       elseif ((fund_freq_lower <= note/3) && (note/3 <= fund_freq_higher))
           music_note = fund_freqs(i);
      
       % Fourth Harmonic
       elseif ((fund_freq_lower <= note/4) && (note/4 <= fund_freq_higher))
           music_note = fund_freqs(i);
      
       % Fifth Harmonic
       elseif ((fund_freq_lower <= note/5) && (note/5 <= fund_freq_higher))
           music_note = fund_freqs(i);
       
       % Sixth Harmonic
       elseif ((fund_freq_lower <= note/6) && (note/6 <= fund_freq_higher))
           music_note = fund_freqs(i);
           
       end
   end
 end