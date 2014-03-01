package org.lemming.interfaces;

import org.lemming.data.Frame;
import org.lemming.data.Localization;
import org.lemming.data.Store;

/**
 * A localizer is transforms Frames into Localizations.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface Localizer<T, F extends Frame<T>> extends Source<Localization>, Well<F> {
}
